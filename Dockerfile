FROM rust

RUN apt-get update
RUN apt-get install -qq cmake liblapack-dev liblapacke-dev

WORKDIR /src/scs
COPY include/ include
COPY linsys/ linsys
COPY src/ src
COPY cmake/ cmake
COPY CMakeLists.txt .
WORKDIR /src/scs/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make