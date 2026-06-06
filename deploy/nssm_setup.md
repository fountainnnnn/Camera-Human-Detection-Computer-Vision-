# NSSM Setup

Create a Windows service with these values:

```text
Service name: SecurityAI
Application path: C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe
Arguments: -NoProfile -ExecutionPolicy Bypass -File C:\security-ai\deploy\run_security_ai.ps1
Startup directory: C:\security-ai
Stdout log: C:\security-ai\logs\nssm_stdout.log
Stderr log: C:\security-ai\logs\nssm_stderr.log
Restart on failure: enabled
Startup type: automatic
```

Recommended steps:

1. Install NSSM.
2. Open an elevated PowerShell or Command Prompt.
3. Run `nssm install SecurityAI`.
4. Fill the fields above.
5. In the I/O tab, set stdout/stderr logs under `logs\`.
6. In the Exit actions tab, enable automatic restart.
7. Start the service with `nssm start SecurityAI`.

Using the PowerShell runner keeps NSSM aligned with Task Scheduler deployment and preserves the same local restart loop and runtime log behavior.
