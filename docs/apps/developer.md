# Developer Guide: Building a New Application

This guide explains how to build, codesign, deploy, and notarize a new application using the `magentart::core` C++ inference engine.

---

## 1. Setting up Git & Pre-Commit Hooks

Before you start developing, configure pre-commit hooks to enforce formatting and licensing:

```bash
# Install pre-commit
uv pip install pre-commit

# Register git hooks in this repository
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

---

## 2. Creating a New Application Target

To create a new application (e.g., `myapp`), you should create a new subdirectory under `examples/` (e.g., `examples/myapp/`) and define its build in a `CMakeLists.txt`.

### Step A: Link against `magentart::core`
Your application's `CMakeLists.txt` needs to link against the portable C++ inference library `magentart::core`. This pulls in MLX, TensorFlow Lite, SentencePiece, and the necessary macOS frameworks.

```cmake
add_executable(myapp MACOSX_BUNDLE
    main.cpp
    # Include other source files
)

target_link_libraries(myapp PRIVATE
    magentart::core
    # Link other frameworks your UI or host needs (e.g., Cocoa, AudioToolbox)
)
```

### Step B: Copy `mlx.metallib` and Codesign Nested Binaries
MLX requires its Metal kernel shaders (`mlx.metallib`) to be colocated with the loading executable/module. You must copy it in a post-build step, and codesign it *before* codesigning the parent bundle.

> **Two patterns in this repo.** Plugin externals (`max`, `pd`, `sc`) do the metallib copy and codesigning in an `add_custom_command(TARGET … POST_BUILD …)` step, as shown below, keeping the separate `deploy_*` target (Step C) just for copying the bundle into place. GUI app bundles (`collider`, `jam`, `standalone`) instead fold the metallib copy, codesigning, *and* deploy into a single `deploy_*` custom target. Either approach works — follow whichever existing app is closest to yours.

```cmake
add_custom_command(TARGET myapp POST_BUILD
    # 1. Copy mlx.metallib next to the binary
    COMMAND ${CMAKE_COMMAND} -E copy
        "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
        "$<TARGET_BUNDLE_DIR:myapp>/Contents/MacOS/mlx.metallib"

    # 2. Codesign the nested metallib first
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_BUNDLE_DIR:myapp>/Contents/MacOS/mlx.metallib"

    # 3. Codesign the parent bundle (use entitlements if needed)
    COMMAND codesign ${CODESIGN_FLAGS}
        # --entitlements "${CMAKE_CURRENT_SOURCE_DIR}/Entitlements.plist" --generate-entitlement-der
        "$<TARGET_BUNDLE_DIR:myapp>"
    COMMENT "Finalizing myapp bundle with codesigning"
    VERBATIM
)
```

### Step C: Create a deploy target
Create a custom target to copy the finalized bundle to `~/Applications` or your desired deployment path.

```cmake
add_custom_target(deploy_myapp
    DEPENDS myapp
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$ENV{HOME}/Applications/MyNewApp.app"
    COMMAND ditto "$<TARGET_BUNDLE_DIR:myapp>" "$ENV{HOME}/Applications/MyNewApp.app"
    COMMENT "Deploying MyNewApp to ~/Applications"
    VERBATIM
)
```

---

## 3. Registering a Notarization Target

The root `CMakeLists.txt` defines a helper function `magentart_add_notarize_target` which packages, uploads, and staples the notarization ticket to your deployed bundle in one shot.

To register your application for notarization, call it at the bottom of your app's `CMakeLists.txt`:

```cmake
magentart_add_notarize_target(notarize_myapp
    APP "$ENV{HOME}/Applications/MyNewApp.app"
    DEPENDS deploy_myapp
    ZIP_NAME "MyNewApp_Dist.zip"
)
```

The function also accepts an optional `EXTRA_FILES` argument — a list of additional files to include alongside the `.app` in the distribution zip (e.g. a README or license):

```cmake
magentart_add_notarize_target(notarize_myapp
    APP "$ENV{HOME}/Applications/MyNewApp.app"
    DEPENDS deploy_myapp
    ZIP_NAME "MyNewApp_Dist.zip"
    EXTRA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
)
```

This will automatically create a `notarize_myapp` target. You can run it directly from your terminal:

```bash
# Compile, deploy, notarize, and package ONLY myapp:
cmake --build build --target notarize_myapp
```

When you run this target, CMake will:
1. Compile and sign the binary and its dependencies.
2. Deploy the app bundle to `~/Applications/`.
3. Compress it into a ZIP archive.
4. Submit the zip to Apple's Notary service using your saved credentials (`notarytool-creds`).
5. Wait for Apple's servers to approve.
6. Staple the approval ticket directly onto the `.app` bundle under `~/Applications/`.
7. Output the final, ready-to-distribute `MyNewApp_Dist.zip` file inside the `build/` directory.

---

## 4. Integrating with build/notarize scripts

To include your new target in the repository-wide automated scripts, update the following files:

1. **`examples/scripts/build-all.sh`**:
   Add your build target to compile it locally:
   ```bash
   build_target "deploy_myapp" "My New App"
   ```

2. **`examples/scripts/notarize-all.sh`**:
   Add your notarize target to submit it to Apple's servers:
   ```bash
   notarize_cmake_target "notarize_myapp" "My New App"
   ```
