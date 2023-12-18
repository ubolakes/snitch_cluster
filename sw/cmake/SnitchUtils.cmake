
function(target_objdump TARGET)
    add_custom_command(TARGET ${TARGET} POST_BUILD
            BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${CMAKE_OBJDUMP} ARGS -D -d ${CMAKE_CURRENT_BINARY_DIR}/${TARGET} > ${TARGET}.d
            COMMENT "Generating objdump ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d"
    )
endfunction()

function(target_dwarfdump TARGET)
    add_custom_command(TARGET ${TARGET} POST_BUILD
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.dwarf
            COMMAND ${CMAKE_DWARFDUMP} ARGS ${TARGET} > ${TARGET}.dwarf
            COMMENT "Generating objdump ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.dwarf"
    )
endfunction()

# Use this to transform lists, e.g. make paths absolute, and export them to the parent scope
macro(transform_lists)
    cmake_parse_arguments(
            MY
            "PARENT_SCOPE;MARK_AS_ADVANCED"
            "PREPEND"
            "LISTS"
            ${ARGN}
    )
    message(DEBUG "transform_lists(LISTS=${MY_LISTS}, PREPEND=${MY_PREPEND}, PARENT_SCOPE=${MY_PARENT_SCOPE}, MARK_AS_ADVANCED=${MARK_AS_ADVANCED})")

    foreach (L IN ITEMS ${MY_LISTS})
        if(DEFINED MY_PREPEND)
            list(TRANSFORM ${L} PREPEND ${MY_PREPEND})
        endif ()
        if("${MY_PARENT_SCOPE}")
            set(${L} ${${L}} PARENT_SCOPE)
        endif ()
        message(DEBUG "${L} = ${${L}}")
    endforeach ()
    if("${MARK_AS_ADVANCED}")
        mark_as_advanced(${MY_LISTS})
    endif ()
endmacro()