# escape=`

# Use a specific tagged image. Tags can be changed, though that is unlikely for most images.
# You could also use the immutable tag @sha256:324e9ab7262331ebb16a4100d0fb1cfb804395a766e3bb1806c62989d1fc1326
# mcr.microsoft.com/dotnet/framework/sdk:4.8-windowsservercore-ltsc2019
ARG FROM_IMAGE=mcr.microsoft.com/windows:2004
FROM ${FROM_IMAGE}

#choco
RUN @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command " [System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
RUN choco install -y 7zip.install git.install
RUN choco install -y activeperl -version 5.24.3.2404001
RUN choco install -y cmake --installargs '"ADD_CMAKE_TO_PATH=System"'
RUN choco install -y visualstudio2019buildtools  --package-parameters "--includeRecommended --includeOptional"
RUN choco install -y visualstudio2019-workload-vctools
RUN choco install -y cuda

#build olminer
RUN git clone https://github.com/overline-mining/olminer.git && ` 
    mkdir C:\olminer\build
RUN cd C:\olminer\ && `
    git submodule update --init --recursive && `
    cd build\ && `
    cmake .. && `
    cmake --build . --config Release --target package
