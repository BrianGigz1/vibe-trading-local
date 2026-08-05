' Launch the daily collection with no visible console window.
'
' Task Scheduler's own "Hidden" flag only suppresses the task's own window; a
' .cmd launched from it still flashes a console. Running it through wscript
' with intWindowStyle 0 keeps it genuinely invisible, so the laptop can run this
' on logon without anything appearing on screen.
'
' bWaitOnReturn = True so the task's exit code reflects the real result rather
' than "launched successfully".
Option Explicit

Dim shell, fso, here, cmd, rc
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = """" & here & "\run_collector.cmd" & """"

rc = shell.Run(cmd, 0, True)
WScript.Quit rc
