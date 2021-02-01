# escape=`
FROM mcr.microsoft.com/windows:2004

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
    cmake --build . --config Release --target install


FROM mcr.microsoft.com/windows/servercore:2004

RUN @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command " [System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
RUN choco install -y vcredist140

RUN mkdir C:\olminer

ARG src="C:\Program Files (x86)\olminer\bin\olminer.exe"
ARG tgt="C:\olminer\"
COPY --from=0 $src $tgt

RUN netsh interface ip set dns name="Ethernet" static 8.8.8.8

#RUN net user /add olminer
#USER olminer

WORKDIR "C:\olminer\"
 
ENTRYPOINT .\\olminer.exe --%MINER_TYPE% -P %STRATUM_URL%