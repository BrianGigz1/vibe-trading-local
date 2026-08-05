' Launch the dashboard server with no visible window.
'
' bWaitOnReturn = False here (unlike the collector's launcher): the dashboard
' runs indefinitely, so waiting would hold the wscript process open forever and
' Task Scheduler would report the task as permanently running.
Option Explicit

Dim shell, fso, here
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
shell.Run """" & here & "\run_dashboard.cmd""", 0, False
