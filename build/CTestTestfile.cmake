# CMake generated Testfile for 
# Source directory: /home/cool777/PCG-CP
# Build directory: /home/cool777/PCG-CP/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(PCGTests "/home/cool777/PCG-CP/build/test_exec")
set_tests_properties(PCGTests PROPERTIES  _BACKTRACE_TRIPLES "/home/cool777/PCG-CP/CMakeLists.txt;64;add_test;/home/cool777/PCG-CP/CMakeLists.txt;0;")
subdirs("googletest")
