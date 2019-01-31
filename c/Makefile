LDFLAGS:=
CXXFLAGS:=-Ofast -march=native
CXX:=clang++

all: bench main.s

bench: main.cpp
	$(CXX) -std=c++17 $(CXXFLAGS) $(LDFLAGS) main.cpp -o bench

main.s: main.cpp
	$(CXX) -std=c++17 $(CXXFLAGS) -S main.cpp

run: bench
	sudo perf stat -e cpu-cycles,instructions,cache-references,cache-misses,branches,branch-misses,bus-cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses,branch-loads,branch-load-misses,cpu-clock,task-clock,page-faults,minor-faults,major-faults,context-switches,cpu-migrations,alignment-faults,emulation-faults ./bench
clean:
	rm -rf bench *.o *.s
