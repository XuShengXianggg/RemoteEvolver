"""YAML runtime options mapped to the inherited core's process configuration."""
RUNTIME_SETTINGS = {
    'tool_call_timeout_seconds': ('EVOAGENT_TOOL_CALL_TIMEOUT', 180, float),
    'mcp_start_timeout_seconds': ('REMOTEEVOLVER_MCP_START_TIMEOUT', 15, float),
    'embedding_timeout_seconds': ('REMOTEEVOLVER_EMBEDDING_TIMEOUT', 180, float),
    'embedding_max_retries': ('REMOTEEVOLVER_EMBEDDING_RETRIES', 1, int),
    'github_api_url': ('REMOTEEVOLVER_GITHUB_API', 'https://api.github.com', str),
    'github_timeout_seconds': ('REMOTEEVOLVER_GITHUB_TIMEOUT', 30, float),
    'clone_install_timeout_seconds': ('REMOTEEVOLVER_CLONE_TIMEOUT', 180, float),
    'skill_compile_attempts': ('REMOTEEVOLVER_SKILL_COMPILE_ATTEMPTS', 2, int),
    'skill_max_bytes': ('REMOTEEVOLVER_SKILL_MAX_BYTES', 8192, int),
    'main_call_attempts': ('REMOTEEVOLVER_MAIN_ATTEMPTS', 2, int),
    'tool_candidate_top_k': ('REMOTEEVOLVER_TOOL_TOP_K', 3, int),
    'tool_repo_attempts': ('REMOTEEVOLVER_TOOL_REPO_ATTEMPTS', 2, int),
    'tool_wrapper_attempts': ('REMOTEEVOLVER_TOOL_WRAPPER_ATTEMPTS', 2, int),
    'tool_smoke_timeout_seconds': ('REMOTEEVOLVER_SMOKE_TIMEOUT', 60, float),
}
