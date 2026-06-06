from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_task_scheduler_script_registers_startup_and_restart_policy() -> None:
    script = (PROJECT_ROOT / "deploy" / "install_task_scheduler.ps1").read_text(encoding="utf-8")

    assert "New-ScheduledTaskTrigger -AtStartup" in script
    assert "RestartCount 999" in script
    assert "RestartInterval (New-TimeSpan -Minutes 1)" in script
    assert "Register-ScheduledTask" in script


def test_runner_script_restarts_after_nonzero_exit_and_stops_on_success() -> None:
    script = (PROJECT_ROOT / "deploy" / "run_security_ai.ps1").read_text(encoding="utf-8")

    assert "while ($true)" in script
    assert "if ($RunOnce -or $exitCode -eq 0)" in script
    assert 'Write-RunnerLog "runtime exited with code $exitCode; runner stopping"' in script
    assert 'Write-RunnerLog "runtime exited with code $exitCode; restarting in $RestartDelaySeconds second(s)"' in script
    assert "Start-Sleep -Seconds $RestartDelaySeconds" in script


def test_nssm_setup_documents_automatic_restart_and_startup() -> None:
    guide = (PROJECT_ROOT / "deploy" / "nssm_setup.md").read_text(encoding="utf-8")

    assert "Restart on failure: enabled" in guide
    assert "Startup type: automatic" in guide
    assert "enable automatic restart" in guide
