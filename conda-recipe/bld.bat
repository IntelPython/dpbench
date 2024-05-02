REM SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
REM
REM SPDX-License-Identifier: Apache-2.0

REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
SET "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

REM Since the 60.0.0 release, setuptools includes a local, vendored copy
REM of distutils (from late copies of CPython) that is enabled by default.
REM It breaks build for Windows, so use distutils from "stdlib" as before.
REM @TODO: remove the setting, once transition to build backend on Windows
REM to cmake is complete.
SET "SETUPTOOLS_USE_DISTUTILS=stdlib"

set "DPBENCH_SYCL=1"
set "CMAKE_GENERATOR=Ninja"
set "CC=icx"
set "CXX=icx"
:: Make CMake verbose
set "VERBOSE=1"

FOR %%V IN (14.0.0 14 15.0.0 15 16.0.0 16 17.0.0 17) DO @(
  REM set DIR_HINT if directory exists
  IF EXIST "%BUILD_PREFIX%\Library\lib\clang\%%V\" (
     SET "SYCL_INCLUDE_DIR_HINT=%BUILD_PREFIX%\Library\lib\clang\%%V"
  )
)

set "PATCHED_CMAKE_VERSION=3.26"
set "PLATFORM_DIR=%PREFIX%\Library\share\cmake-%PATCHED_CMAKE_VERSION%\Modules\Platform"
set "FN=Windows-IntelLLVM.cmake"

rem Save the original file, and copy patched file to
rem fix the issue with IntelLLVM integration with cmake on Windows
if EXIST "%PLATFORM_DIR%" (
  dir "%PLATFORM_DIR%\%FN%"
  copy /Y "%PLATFORM_DIR%\%FN%" .
  if errorlevel 1 exit 1
  copy /Y ".github\workflows\Windows-IntelLLVM_%PATCHED_CMAKE_VERSION%.cmake" "%PLATFORM_DIR%\%FN%"
  if errorlevel 1 exit 1
)

:: -wnx flags mean: --wheel --no-isolation --skip-dependency-check
%PYTHON% -m build -w -n -x
if %ERRORLEVEL% neq 0 exit 1

:: `pip install dist\dpbench*.whl` does not work on windows,
:: so use a loop; there's only one wheel in dist/ anyway
for /f %%f in ('dir /b /S .\dist') do (
    %PYTHON% -m wheel tags --remove --build %GIT_DESCRIBE_NUMBER% %%f
    if %ERRORLEVEL% neq 0 exit 1
)

:: wheel file was renamed
for /f %%f in ('dir /b /S .\dist') do (
    %PYTHON% -m pip install %%f ^
      --no-build-isolation ^
      --no-deps ^
      --only-binary :all: ^
      --no-index ^
      --prefix %PREFIX% ^
      -vv
    if %ERRORLEVEL% neq 0 exit 1
)

:: Must be consistent with pyproject.toml project.scritps. Currently pip does
:: not allow to ignore scripts installation, so we have to remove them manually.
:: https://github.com/pypa/pip/issues/3980
:: We have to let conda-build manage it for use in order to set proper python
:: path.
:: https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html#python-entry-points
rm %PREFIX%\Scripts\dpbench.exe

:: Copy wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    copy dist\dpbench*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)

rem copy back
if EXIST "%PLATFORM_DIR%" (
   copy /Y "%FN%" "%PLATFORM_DIR%\%FN%"
   if errorlevel 1 exit 1
)
