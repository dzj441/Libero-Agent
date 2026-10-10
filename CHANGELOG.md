# Changelog

## 0.1.0
- publish main 30 evalset
- Use standard system EGL by default; retain explicitly configured driver
  bundles for machines that need them.
- Restrict native Codex filesystem permissions to the agent workspace,
  minimal system tools, and the installed CLI executable required to start
  its sandbox. External harness isolation remains an explicit limitation.
