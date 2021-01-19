set(SOURCES
	EthashAux.h EthashAux.cpp
	Farm.cpp Farm.h
	Miner.h Miner.cpp
)

include_directories(BEFORE ..)

add_library(ethcore ${SOURCES})
target_link_libraries(ethcore PUBLIC devcore ethash::ethash PRIVATE hwmon)

if(ETHASHCL)
	target_link_libraries(ethcore PRIVATE ethash-cl)
endif()
if(OLHASHCUDA)
	target_link_libraries(ethcore PUBLIC olhash-cuda)
endif()
if(OLHASHCPU)
	target_link_libraries(ethcore PUBLIC olhash-cpu)
endif()
