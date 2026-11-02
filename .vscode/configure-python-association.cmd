@echo off
setlocal
set "taskResult=%~dp0python-association-admin-result.txt"
> "%taskResult%" echo Configuring Python file association...
>> "%taskResult%" 2>&1 ftype Python.File="C:\ProgramData\miniconda3\python.exe" "%%1" %%*
if errorlevel 1 goto failed
>> "%taskResult%" 2>&1 assoc .py=Python.File
if errorlevel 1 goto failed
>> "%taskResult%" echo SUCCESS
exit /b 0
:failed
>> "%taskResult%" echo FAILED
exit /b 1
