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

# Standalone macOS app (MRT2) using magentart::core.
#
# Resources (musiccoca, spectrostream) are loaded from ~/Documents/Magenta/resources/
# at runtime — see examples/shared/cpp/magenta_paths.h.

set(APP_ICON "${CMAKE_CURRENT_SOURCE_DIR}/../../common/assets/AppIcon.icns")
set_source_files_properties(${APP_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

add_executable(mrt2_standalone MACOSX_BUNDLE
    MagentaRTStandaloneApp.mm
    MagentaRTAppController.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cpp/magenta_paths.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/objc/MagentaModelManager.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/objc/MagentaModelDownloader.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/objc/MagentaSettings.mm
    ${APP_ICON}
)

target_include_directories(mrt2_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/objc
)
target_compile_options(mrt2_standalone PRIVATE -fobjc-arc)

target_link_libraries(mrt2_standalone PRIVATE
    magentart::core
    "-framework AppKit"
    "-framework AudioToolbox"
    "-framework AVFoundation"
    "-framework CoreAudio"
    "-framework CoreMIDI"
    "-framework WebKit"
    "-framework UniformTypeIdentifiers"
)

set_target_properties(mrt2_standalone PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/StandaloneInfo.plist.in"
    MACOSX_BUNDLE_EXECUTABLE_NAME MRT2
    MACOSX_BUNDLE_BUNDLE_NAME "MRT2"
    MACOSX_BUNDLE_ICON_FILE "AppIcon.icns"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.google.mrt2.standalone"
)

add_dependencies(mrt2_standalone build_mrt2_ui)

# ─── Deploy standalone app ───────────────────────────────────────────────────
add_custom_target(deploy_mrt2_standalone
    DEPENDS mrt2_standalone
    # Copy React UI dist into app Resources
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_DIR:mrt2_standalone>/Contents/Resources/ui"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../react_ui/dist"
        "$<TARGET_BUNDLE_DIR:mrt2_standalone>/Contents/Resources/ui"
    # Copy mlx.metallib next to the binary
    COMMAND ${CMAKE_COMMAND} -E copy
        "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
        "$<TARGET_BUNDLE_DIR:mrt2_standalone>/Contents/MacOS/mlx.metallib"
    # Code sign
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_BUNDLE_DIR:mrt2_standalone>/Contents/MacOS/mlx.metallib"
    COMMAND codesign ${CODESIGN_FLAGS}
        --entitlements "${CMAKE_CURRENT_SOURCE_DIR}/StandaloneEntitlements.plist"
        --generate-entitlement-der
        "$<TARGET_BUNDLE_DIR:mrt2_standalone>"
    # Deploy to ~/Applications
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        "$ENV{HOME}/Applications/MRT2.app"
    COMMAND ditto
        "$<TARGET_BUNDLE_DIR:mrt2_standalone>"
        "$ENV{HOME}/Applications/MRT2.app"
    COMMENT "Building and deploying MRT2 standalone app"
    VERBATIM
)

# Notarize the deployed MRT2.app (requires Developer ID).
magentart_add_notarize_target(notarize_mrt2_standalone
    APP "$ENV{HOME}/Applications/MRT2.app"
    DEPENDS deploy_mrt2_standalone
    ZIP_NAME "MRT2_Standalone.zip"
)
