# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ─── AUv3 Extension (.appex) — MRT2 ─────────────────────────────────────────────────
set(APP_ICON "${CMAKE_CURRENT_SOURCE_DIR}/../../common/assets/AppIcon.icns")
set_source_files_properties(${APP_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

add_executable(mrt2_au MACOSX_BUNDLE
    MagentaRT_AudioUnit.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cpp/magenta_paths.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/objc/MagentaModelManager.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/objc/MagentaModelDownloader.mm
    ${APP_ICON}
)

# Link NSExtensionMain as the entry point (no user-written main())
target_link_options(mrt2_au PRIVATE -e _NSExtensionMain)

target_include_directories(mrt2_au PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/objc
)
target_compile_options(mrt2_au PRIVATE -fobjc-arc)

target_link_libraries(mrt2_au PRIVATE
    magentart::core
    "-framework AppKit"
    "-framework AudioToolbox"
    "-framework AVFoundation"
    "-framework CoreAudioKit"
    "-framework WebKit"
    "-framework UniformTypeIdentifiers"
)

set_target_properties(mrt2_au PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
    MACOSX_BUNDLE_EXECUTABLE_NAME MRT2_AU
    MACOSX_BUNDLE_BUNDLE_NAME "MRT2_AU"
    MACOSX_BUNDLE_ICON_FILE "AppIcon.icns"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.google.mrt2.au"
)

# ─── AUv3 Host App (.app) ─────────────────────────────────────────────────────
add_executable(mrt2_au_app MACOSX_BUNDLE
    MagentaRT_AUHostApp.mm
    ${APP_ICON}
)

target_compile_options(mrt2_au_app PRIVATE -fobjc-arc)

target_link_libraries(mrt2_au_app PRIVATE
    "-framework Cocoa"
)

set_target_properties(mrt2_au_app PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/HostInfo.plist.in"
    MACOSX_BUNDLE_EXECUTABLE_NAME MRT2_AUApp
    MACOSX_BUNDLE_BUNDLE_NAME "MRT2 (AU)"
    MACOSX_BUNDLE_ICON_FILE "AppIcon.icns"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.google.mrt2.au.host"
)



# ─── Embed, sign, and deploy — always runs via custom target ──────────────────
add_dependencies(mrt2_au_app mrt2_au)
add_dependencies(mrt2_au_app build_mrt2_ui)

set(RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
set(EXTERNAL_RESOURCE_DIR "$ENV{HOME}/Documents/Magenta/magenta-rt-v2/resources")
set(ENTITLEMENTS "${CMAKE_CURRENT_SOURCE_DIR}/Entitlements.plist")

add_custom_target(deploy_mrt2_au ALL
    DEPENDS mrt2_au_app mrt2_au
    # Re-embed appex into host app (always, not just on re-link)
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns"
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<TARGET_BUNDLE_DIR:mrt2_au>"
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex"
    # Create appex Resources folder
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex/Contents/Resources"
    # Copy React UI dist into appex Resources
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex/Contents/Resources/ui"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../react_ui/dist"
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex/Contents/Resources/ui"
    # Copy mlx.metallib next to the AU binary (MLX runtime resolution: colocated search)
    COMMAND ${CMAKE_COMMAND} -E copy
        "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex/Contents/MacOS/mlx.metallib"
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex/Contents/MacOS/mlx.metallib"
    # Code sign
    COMMAND codesign ${CODESIGN_FLAGS}
        --entitlements "${ENTITLEMENTS}" --generate-entitlement-der
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>/Contents/PlugIns/MRT2_AU.appex"
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>"
    # Deploy to ~/Applications
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        "$ENV{HOME}/Applications/MRT2 (AU).app"
    COMMAND ditto
        "$<TARGET_BUNDLE_DIR:mrt2_au_app>"
        "$ENV{HOME}/Applications/MRT2 (AU).app"
    # Kill registrar first so it restarts fresh, then register
    COMMAND sh -c "killall -9 AudioComponentRegistrar 2>/dev/null || true"
    COMMAND sleep 1
    COMMAND sh -c "pluginkit -a '$ENV{HOME}/Applications/MRT2 (AU).app/Contents/PlugIns/MRT2_AU.appex' 2>/dev/null || true"
    COMMENT "Embedding, signing, and deploying MRT2 (AU).app"
    VERBATIM
)

# Notarize the deployed MRT2 (AU).app (requires Developer ID).
magentart_add_notarize_target(notarize_mrt2_au
    APP "$ENV{HOME}/Applications/MRT2 (AU).app"
    DEPENDS deploy_mrt2_au
    ZIP_NAME "MRT2_AU.zip"
    EXTRA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/INSTALL.md"
)
