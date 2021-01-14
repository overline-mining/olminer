cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

set(EXECUTABLE olminer)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
if(MSVC)
	target_sources(${EXECUTABLE} PRIVATE olminer.rc)
endif()

hunter_add_package(CLI11)
find_package(CLI11 CONFIG REQUIRED)

target_link_libraries(olminer PRIVATE ethcore poolprotocols devcore olminer-buildinfo CLI11::CLI11 Boost::system Boost::thread)

if(ETHDBUS)
	find_package(PkgConfig)
	set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/lib/x86_64-linux-gnu/pkgconfig" )
	pkg_check_modules(DBUS dbus-1)
	include_directories(${DBUS_INCLUDE_DIRS})
	link_directories(${DBUS_LIBRARY_DIRS})
	target_link_libraries(olminer PRIVATE ${DBUS_LIBRARIES})
endif()

if(APICORE)
	target_link_libraries(olminer PRIVATE apicore)
endif()

include(GNUInstallDirs)
install(TARGETS olminer DESTINATION ${CMAKE_INSTALL_BINDIR})
if(MSVC)
    install(FILES $<TARGET_PDB_FILE:olminer> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
endif()
