# Detect if the install is run by CPack.
if (${CMAKE_INSTALL_PREFIX} MATCHES "/_CPack_Packages/.*/(TGZ|ZIP)/")
    # Flatten the directory structure such that everything except the header files is placed in root.
    file(GLOB bin_files LIST_DIRECTORIES FALSE ${CMAKE_INSTALL_PREFIX}/bin/*)
    file(GLOB lib_files LIST_DIRECTORIES FALSE ${CMAKE_INSTALL_PREFIX}/lib/*)
    file(GLOB data_files LIST_DIRECTORIES FALSE ${CMAKE_INSTALL_PREFIX}/share/*)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/olminer)
    foreach(file ${bin_files} ${lib_files} ${data_files})
        get_filename_component(file_name ${file} NAME)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E rename
            ${file}
            ${CMAKE_INSTALL_PREFIX}/olminer/${file_name}
        )
    endforeach()
    execute_process( COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/bin)
    execute_process( COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/lib)
    execute_process( COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/share)
endif()
