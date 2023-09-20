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

"%PYTHON%" setup.py clean --all

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

@REM TODO: switch to pip build. Currently results in broken binary
@REM %PYTHON% -m pip install --no-index --no-deps --no-build-isolation . -v
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    rem Install and assemble wheel package from the build bits
    "%PYTHON%" setup.py install bdist_wheel --single-version-externally-managed --record=record.txt
    if errorlevel 1 exit 1
    copy dist\dpbench*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
) ELSE (
    rem Only install
    "%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
    if errorlevel 1 exit 1
)

rem copy back
if EXIST "%PLATFORM_DIR%" (
   copy /Y "%FN%" "%PLATFORM_DIR%\%FN%"
   if errorlevel 1 exit 1
)
