# Example Build and Distribution Scripts

This directory contains shell scripts to automate building, signing, and notarizing the MRT projects on macOS.

For full details on the distribution guidelines, ad-hoc signing vs Developer ID signing, and manual Gatekeeper troubleshooting, see [docs/apps/distributing.md](../../docs/apps/distributing.md).

---

## 1. Local Development Build: `build-all.sh`

This script compiles, codesigns, and deploys all projects, audio host externals, and CLI programs to their respective local folders.

Use this script during local development to quickly compile changes and test them.

### Usage

```bash
bash examples/scripts/build-all.sh
```

### Targets Built & Deployed

* **macOS Applications** (deployed to `~/Applications`):
  * `MRT2.app` (Standalone)
  * `MRT2 (AU).app` (AUv3 Plugin Host)
  * `Jam.app`
  * `MRT2 - Collider.app`
* **Audio Host Externals**:
  * **Max MSP**: `mrt~.mxo` (deployed to `~/Documents/Max 9/Library`)
  * **Pure Data**: `mrt~.pd_darwin` (deployed to `~/Documents/Pd/externals/mrt~`)
  * **SuperCollider**: `MRT2.scx` (deployed to `~/Library/Application Support/SuperCollider/Extensions/MRT2`)
* **CLI Executables** (built inside `build/examples/`):
  * `hello_mrt2`

---

## 2. Release Distribution & Notarization: `notarize-all.sh`

This script builds, signs with Hardened Runtime, and submits all projects to Apple's Notary service to prepare them for distribution to other macOS users. Once notarization is complete, it packages them into ready-to-distribute `.zip` archives.

> [!IMPORTANT]
> This process requires a paid Apple Developer Program membership and network connectivity.

### Prerequisites

1. Create a **Developer ID Application** certificate in Xcode (Settings → Accounts → Manage Certificates).
2. Configure CMake with your identity by passing `-DCODESIGN_IDENTITY="..."` or exporting `MAGENTART_DEVELOPER_ID`:
   ```bash
   export MAGENTART_DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)"
   ```
3. Create an app-specific password for Apple's notary service and save it to your keychain:
   ```bash
   xcrun notarytool store-credentials "notarytool-creds" \
       --apple-id "<Your-Apple-ID>" \
       --team-id <Your-Team-ID> \
       --password <Your-App-Specific-Password>
   ```

### Usage

```bash
bash examples/scripts/notarize-all.sh [--keychain-profile <profile-name>]
```

* `--keychain-profile`: *(Optional)* The name of the keychain credentials profile to use for notarization. Defaults to `"notarytool-creds"`.

### Outputs

Upon completion, separate, notarized `.zip` files will be output to your `build/` directory:
* `MRT2_AU.zip` (AUv3 Plugin Host)
* `MRT2_Standalone.zip` (Standalone version of AUv3)
* `MRT2_Jam.zip` (Jam App)
* `MRT2_Collider.zip` (Collider App)
* `MRT2_Max.zip` (Max MSP External)
* `MRT2_Pd.zip` (Pure Data External)
* `MRT2_SuperCollider.zip` (SuperCollider UGen)
