"""Coordinator for frozen-selection activation and runtime checks."""

from __future__ import annotations

import json
from pathlib import Path
from ...infra.paths import runtime_path
from typing import Any, Mapping

from .audit_log import append_tool_validation_log
from .capability_contract import CapabilityContract, fingerprint_json
from .runtime_contract import (
    RuntimeContractResult,
    check_invocation_postconditions,
    check_invocation_preconditions,
)
from .tool_certification import (
    CertificationStore,
    check_certification,
)


DEFAULT_DECLARATIONS = runtime_path("tools/_capability_contracts.json")


class SelectedToolVerificationError(RuntimeError):
    """A frozen tool selection cannot be activated in the current MCP process."""

    def __init__(
        self,
        failures: Mapping[str, str],
        *,
        categories: Mapping[str, str] | None = None,
    ) -> None:
        self.failures = dict(failures)
        self.categories = dict(categories or {})
        details = "; ".join(f"{name}: {reason}" for name, reason in self.failures.items())
        super().__init__(f"selected tool verification failed: {details}")


class ToolReliabilityManager:
    def __init__(
        self,
        *,
        mode: str = "strict",
        certification_dir: Path | str = runtime_path("tools/.certifications"),
        declarations_path: Path | str = DEFAULT_DECLARATIONS,
    ) -> None:
        if mode not in {"strict", "permissive"}:
            raise ValueError("tool verification mode must be strict or permissive")
        self.mode = mode
        self.store = CertificationStore(certification_dir)
        self.declarations_path = Path(declarations_path)
        self._contracts = self._load_declarations()
        self._active: dict[str, CapabilityContract] = {}
        self._required: dict[str, CapabilityContract] = {}
        self._active_schema_fingerprints: dict[str, str] = {}
        self._covered_requirement_fingerprints: set[str] = set()

    @property
    def active_contracts(self) -> dict[str, CapabilityContract]:
        return dict(self._active)

    @property
    def covered_requirement_fingerprints(self) -> set[str]:
        return set(self._covered_requirement_fingerprints)

    def contract_for(self, tool_name: str) -> CapabilityContract | None:
        """Expose the selected tool contract to the independent interface agent."""
        return self._active.get(tool_name) or self._contract_for(tool_name.split("__")[-1])

    def requirement_for(self, tool_name: str) -> CapabilityContract | None:
        """Expose the task capability matched to a selected tool."""
        return self._required.get(tool_name)

    def offered_contracts(
        self, schemas: list[dict[str, Any]],
    ) -> dict[str, CapabilityContract]:
        """Return semantic declarations for planning without certifying tools."""
        output: dict[str, CapabilityContract] = {}
        for schema in schemas:
            qualified = str(schema.get("function", {}).get("name", schema.get("name", "")))
            local_name = qualified.split("__")[-1]
            contract = self._contract_for(local_name)
            if qualified and contract is not None:
                output[qualified] = contract
        return output

    def eligible_schemas_for_react(
        self,
        schemas: list[dict[str, Any]],
        *,
        task_id: str = "",
        task: str = "",
    ) -> list[dict[str, Any]]:
        """Return registered tools that are engineering-safe to expose.

        This is deliberately task-independent.  It checks that an MCP-visible
        schema still matches a maintained declaration or a generated tool's
        certification record; it never compares the tool with the current
        task, an operation, or a frozen capability plan.
        """
        accepted: list[dict[str, Any]] = []
        for schema in schemas:
            qualified = str(
                schema.get("function", {}).get("name", schema.get("name", ""))
            )
            if not qualified:
                continue
            local_name = qualified.split("__")[-1]
            contract = self._contract_for(local_name)
            if contract is None:
                self._audit(
                    task_id, task, qualified, None, None,
                    "engineering_ineligible", "tool has no registration declaration or certification",
                )
                continue
            valid, reason = self._ensure_certified(
                local_name, qualified, schema, contract,
            )
            if not valid:
                self._audit(
                    task_id, task, qualified, contract, None,
                    "engineering_ineligible", reason[1],
                )
                continue
            accepted.append(schema)
            self._audit(
                task_id, task, qualified, contract, None,
                "react_engineering_eligible", reason[1],
            )
        return accepted

    def certify_declared_tool(
        self,
        schema: dict[str, Any],
        *,
        task_id: str = "",
        task: str = "",
    ) -> bool:
        """Validate one declared implementation before selection commits to it."""
        qualified = str(schema.get("function", {}).get("name", schema.get("name", "")))
        local_name = qualified.split("__")[-1]
        offered = self._contract_for(local_name)
        if offered is None:
            self._audit(
                task_id, task, qualified, None, None,
                "insufficient_capability_evidence", "no declared contract",
            )
            return self.mode == "permissive"
        valid, reason = self._ensure_certified(local_name, qualified, schema, offered)
        if not valid:
            self._audit(task_id, task, qualified, offered, None, reason[0], reason[1])
        return valid or self.mode == "permissive"

    def prepare_tools(
        self,
        schemas: list[dict[str, Any]],
        *,
        selected_requirements: Mapping[str, CapabilityContract],
        task_id: str = "",
        task: str = "",
    ) -> list[dict[str, Any]]:
        """Verify and activate an already-frozen selection.

        This method deliberately has no capability matching or candidate-choice
        path.  ``selected_requirements`` is authoritative: every named tool must
        be present and engineering-verifiable before strict-mode activation can
        commit.  It never substitutes a different tool or edits the mapping.
        """
        schemas_by_name = {
            str(schema.get("function", {}).get("name", schema.get("name", ""))): schema
            for schema in schemas
        }
        accepted: list[dict[str, Any]] = []
        active: dict[str, CapabilityContract] = {}
        required: dict[str, CapabilityContract] = {}
        schema_fingerprints: dict[str, str] = {}
        covered: set[str] = set()
        failures: dict[str, str] = {}
        failure_categories: dict[str, str] = {}

        for qualified, requirement in selected_requirements.items():
            schema = schemas_by_name.get(qualified)
            if schema is None:
                failures[qualified] = "selected tool is not visible in current MCP schemas"
                failure_categories[qualified] = "mcp_not_discovered"
                self._audit(
                    task_id, task, qualified, None, None,
                    "selected_tool_not_discovered", failures[qualified],
                )
                continue
            local_name = qualified.split("__")[-1]
            offered = self._contract_for(local_name)
            if offered is None:
                failures[qualified] = "selected tool has no declared capability contract"
                failure_categories[qualified] = "capability_declaration_missing"
                self._audit(
                    task_id, task, qualified, None, None,
                    "insufficient_capability_evidence", failures[qualified],
                )
                continue
            valid, reason = self._ensure_certified(local_name, qualified, schema, offered)
            if not valid and self.mode == "strict":
                failures[qualified] = reason[1]
                failure_categories[qualified] = reason[0]
                self._audit(task_id, task, qualified, offered, None, reason[0], reason[1])
                continue
            if not valid:
                self._audit(task_id, task, qualified, offered, None, reason[0], reason[1], warning=True)
            accepted.append(schema)
            active[qualified] = offered
            required[qualified] = requirement
            schema_fingerprints[qualified] = fingerprint_json(schema)
            covered.add(requirement.fingerprint)
            self._audit(
                task_id, task, qualified, offered, None,
                "selected_frozen", reason[1], warning=not valid,
            )

        if failures:
            self._active.clear()
            self._required.clear()
            self._active_schema_fingerprints.clear()
            self._covered_requirement_fingerprints.clear()
            raise SelectedToolVerificationError(
                failures, categories=failure_categories,
            )

        self._active = active
        self._required = required
        self._active_schema_fingerprints = schema_fingerprints
        self._covered_requirement_fingerprints = covered
        return accepted

    def check_active_selection(
        self,
        schemas: list[dict[str, Any]],
        *,
        selected_tool_names: list[str],
    ) -> list[dict[str, Any]]:
        """Lightweight ReAct-boundary check with no certification or matching.

        The method is read-only: it confirms that the frozen tools are still
        MCP-visible, were activated earlier, and expose the same schemas that
        were activated.  Any inconsistency is explicit and all-or-nothing.
        """
        schemas_by_name = {
            str(schema.get("function", {}).get("name", schema.get("name", ""))): schema
            for schema in schemas
        }
        ordered_names = list(dict.fromkeys(selected_tool_names))
        failures: dict[str, str] = {}
        failure_categories: dict[str, str] = {}
        for qualified in ordered_names:
            schema = schemas_by_name.get(qualified)
            if schema is None:
                failures[qualified] = "selected tool is no longer visible in MCP schemas"
                failure_categories[qualified] = "mcp_not_discovered"
            elif qualified not in self._active:
                failures[qualified] = "selected tool was not activated before ReAct"
                failure_categories[qualified] = "not_activated"
            elif self._active_schema_fingerprints.get(qualified) != fingerprint_json(schema):
                failures[qualified] = "selected tool schema changed after activation"
                failure_categories[qualified] = "schema_changed"
        unexpected_active = set(self._active) - set(ordered_names)
        for qualified in sorted(unexpected_active):
            failures[qualified] = "active tool is not present in the frozen selection"
            failure_categories[qualified] = "unexpected_active_tool"
        if failures:
            raise SelectedToolVerificationError(
                failures, categories=failure_categories,
            )
        return [schemas_by_name[name] for name in ordered_names]

    def revalidate_selected_tool(
        self,
        qualified: str,
        schema: dict[str, Any],
    ) -> tuple[bool, str]:
        """Repeat engineering verification for the same frozen tool.

        Revalidation never changes capability selection.  It is available only
        when the previous verified record retained the exact smoke/probe
        arguments; otherwise regenerating arguments would introduce a new LLM
        decision at recovery time.
        """
        from .invariants import check_output_invariants
        from .registration_probe import probe_registered_tool
        from .test_runner import run_smoke_test
        from .tool_certification import CertificationRecord

        local_name = qualified.split("__")[-1]
        source_path = runtime_path("tools") / f"{local_name}.py"
        if not source_path.is_file():
            return False, "tool source is missing"
        previous = self.store.load(local_name)
        if previous is None:
            return False, "previous certification is missing"
        arguments = previous.evidence.get("probe_arguments")
        if not isinstance(arguments, dict):
            return False, "certification has no deterministic probe arguments"
        try:
            contract = CapabilityContract.from_dict(previous.capability_contract)
        except (TypeError, ValueError):
            return False, "certification capability declaration is invalid"

        code = source_path.read_text(encoding="utf-8")
        smoke = run_smoke_test(code, arguments)
        if not smoke.success:
            return False, f"smoke revalidation failed at {smoke.stage}"
        violations = check_output_invariants(arguments, smoke.output, known_inputs=[])
        if violations:
            return False, f"engineering invariant revalidation failed: {violations[0]}"
        try:
            probe = probe_registered_tool(local_name, arguments)
        except Exception as exc:  # noqa: BLE001
            return False, f"fresh MCP probe failed: {type(exc).__name__}: {exc}"
        verified = CertificationRecord.create(
            tool_name=local_name,
            tool_code=code,
            wrapper_code=code,
            schema=probe["schema"],
            contract=contract,
            smoke_passed=True,
            invariants_passed=True,
            probe_passed=True,
            evidence={
                "smoke_output": smoke.output[:1000],
                "probe_arguments": dict(arguments),
                "mcp_probe": {
                    "qualified_name": probe["qualified_name"],
                    "result": str(probe["result"])[:1000],
                },
            },
        )
        self.store.save(verified)
        return True, "same selected tool passed repeated engineering verification"

    def before_call(
        self, tool_name: str, arguments: dict[str, Any], *, adapters: dict[str, str] | None = None,
    ) -> RuntimeContractResult:
        contract = self.contract_for(tool_name)
        if contract is None:
            if self.mode == "permissive":
                return RuntimeContractResult(True, adapted_arguments=dict(arguments), evidence={"warning": "unverified tool"})
            return RuntimeContractResult(False, "insufficient_capability_evidence", ["tool has no active matched capability contract"], dict(arguments))
        result = check_invocation_preconditions(
            contract, arguments, adapters=adapters, enforce_interface=False,
        )
        self._audit_runtime(tool_name, "pre_call", result)
        if result.passed:
            return result
        return RuntimeContractResult(
            True,
            adapted_arguments=dict(result.adapted_arguments or arguments),
            evidence={**result.evidence, "contract_warnings": list(result.violations)},
        )

    def after_call(
        self, tool_name: str, result_repr: str, *, arguments: dict[str, Any],
    ) -> RuntimeContractResult:
        contract = self.contract_for(tool_name)
        if contract is None:
            if self.mode == "permissive":
                return RuntimeContractResult(True, evidence={"warning": "unverified tool"})
            return RuntimeContractResult(False, "insufficient_capability_evidence", ["tool has no active matched capability contract"])
        result = check_invocation_postconditions(
            contract, result_repr, arguments=arguments, enforce_interface=False,
        )
        self._audit_runtime(tool_name, "post_call", result)
        if result.passed:
            return result
        return RuntimeContractResult(
            True,
            adapted_arguments=dict(arguments),
            evidence={**result.evidence, "contract_warnings": list(result.violations)},
        )

    def _ensure_certified(
        self, local_name: str, qualified: str, schema: dict[str, Any], contract: CapabilityContract,
    ) -> tuple[bool, tuple[str, str]]:
        source_path = runtime_path("tools") / f"{local_name}.py"
        if not source_path.is_file():
            return False, ("tool_source_missing", "remote/non-local tool has no certifiable source")
        code = source_path.read_text(encoding="utf-8")
        record = self.store.load(local_name)
        cached = check_certification(
            record, tool_code=code, wrapper_code=code, schema=schema, required_contract=contract,
        )
        if cached.valid:
            append_tool_validation_log(
                task_id="", task="", stage="certification", candidate_tool=qualified,
                capability_contract=contract.to_dict(), match_result={}, decision="reuse",
                details={
                    "reason": cached.reason,
                    "contract_fingerprint": record.contract_fingerprint if record else "",
                    "tool_fingerprint": record.tool_fingerprint if record else "",
                    "schema_fingerprint": record.schema_fingerprint if record else "",
                },
            )
            return True, ("cached", cached.reason)
        if local_name in self._contracts:
            # Static declarations are maintained source-of-truth entries for
            # built-in tools.  Generated tools must carry their own engineering
            # verification record and never inherit this exception implicitly.
            return True, ("maintained_declaration", "built-in tool has a maintained capability declaration")
        stale_reasons = {
            "tool code changed", "wrapper changed", "schema changed",
            "relevant environment changed",
        }
        category = (
            "engineering_verification_stale"
            if cached.reason in stale_reasons
            else "engineering_verification_missing"
        )
        return False, (category, cached.reason)

    def _contract_for(self, local_name: str) -> CapabilityContract | None:
        contract = self._contracts.get(local_name)
        if contract is not None:
            return contract
        record = self.store.load(local_name)
        if record is None:
            return None
        try:
            return CapabilityContract.from_dict(record.capability_contract)
        except (TypeError, ValueError):
            return None

    def _load_declarations(self) -> dict[str, CapabilityContract]:
        if not self.declarations_path.exists():
            return {}
        value = json.loads(self.declarations_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("tool capability declaration file must be an object")
        return {str(name): CapabilityContract.from_dict(contract) for name, contract in value.items()}

    def _audit(
        self, task_id: str, task: str, tool: str, contract: CapabilityContract | None,
        match: Any | None, decision: str, reason: str, *, warning: bool = False,
    ) -> None:
        append_tool_validation_log(
            task_id=task_id, task=task, stage="selection", candidate_tool=tool,
            capability_contract=contract.to_dict() if contract else {},
            match_result=match.to_dict() if match else {}, decision=decision,
            details={"reason": reason, "warning": warning, "mode": self.mode},
        )

    def _audit_runtime(self, tool: str, stage: str, result: RuntimeContractResult) -> None:
        append_tool_validation_log(
            task_id="", task="", stage=stage, candidate_tool=tool,
            capability_contract=(self._active.get(tool).to_dict() if tool in self._active else {}),
            match_result={}, decision="passed" if result.passed else result.category,
            details=result.to_dict(),
        )
