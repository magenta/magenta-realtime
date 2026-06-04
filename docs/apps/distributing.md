# Distributing the macOS apps

By default, every `deploy_*` target signs the built bundle with an ad-hoc
signature (`CODESIGN_IDENTITY=-`). That's fine on your own machine, but macOS
Gatekeeper will block ad-hoc bundles from launching anywhere else.

The distributable apps — MRT AUv3, MRT Standalone, Jam, and
Collider — share the same workflow. Pick one of the two paths below.

## Ad-hoc (free, requires recipient cooperation)

You can zip the bundle from `~/Applications/` and send it to a colleague,
but they must clear quarantine and re-sign locally before macOS will run it:

```bash
xattr -cr ~/Applications/<App>.app
codesign --force --sign - "$(find ~/Applications/<App>.app -name mlx.metallib)"
codesign --force --sign - ~/Applications/<App>.app
```

For the AUv3, the recipient also needs to register the plug-in extension:

```bash
pluginkit -a ~/Applications/MRT2\ \(AU\).app/Contents/PlugIns/MRT2_AU.appex
```

## Developer ID (notarized, recommended for distribution)

Requires a paid Apple Developer Program membership. One-time setup:

1. In Xcode → Settings → Accounts → Manage Certificates, click the **`+`** button and create a **Developer ID Application** certificate.
2. Create an [app-specific password](https://support.apple.com/en-us/102654) for `notarytool` on [appleid.apple.com](https://appleid.apple.com).
3. Store the credentials in your keychain:
   ```bash
   xcrun notarytool store-credentials "notarytool-creds" \
       --apple-id "<AppleID>" --team-id <TEAMID> --password <app-specific-password>
   ```
   The profile name (`notarytool-creds`) is the default; override with `-DNOTARYTOOL_KEYCHAIN_PROFILE=<name>` if you've used a different one.

### 1. Build & Notarize Everything (All-in-One Script)

For an automated script that compiles, signs, notarizes, and packages all applications, AUv3 plugins, and audio host externals (Max MSP, Pure Data, SuperCollider) in one go:

```bash
# A. Look up your Developer ID Application identity
security find-identity -v -p codesigning

# B. Export the identity to your environment
export MAGENTART_DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)"

# C. Configure and run the script
uv run cmake . -B build
bash examples/scripts/notarize-all.sh
```

All finalized `.zip` files will land in your `build/` directory.

### 2. Build & Notarize Specific Applications

If you only want to notarize a specific application, set up your identity and use individual CMake targets:

```bash
# A. Export your Developer ID Application identity
export MAGENTART_DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)"

# B. Configure the build directory
uv run cmake . -B build

# C. Build, sign, and notarize the target:
uv run cmake --build build --target notarize_mrt2_au          # AUv3 Plugin
uv run cmake --build build --target notarize_mrt2_standalone  # Standalone App
uv run cmake --build build --target notarize_mrt2_jam         # Jam App
uv run cmake --build build --target notarize_mrt2_collider    # Collider App
```

*(Note: If you want to pin a specific identity to a single build folder without exporting an environment variable, pass `-DCODESIGN_IDENTITY="..."` to CMake instead.)*

Each `notarize_*` target automatically runs its corresponding `deploy_*` target to compile, sign (with hardened runtime + secure timestamp), submit the archive, and staple the approved ticket. The distributable zip will be generated in `build/` (e.g., `build/MRT2_AU.zip`).
