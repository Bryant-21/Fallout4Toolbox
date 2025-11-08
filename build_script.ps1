# Start the timer
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "Starting build process..." -ForegroundColor Cyan

# Clean previous build output
Write-Host "`nCleaning previous build..." -ForegroundColor Yellow
if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force }
if (Test-Path "build") { Remove-Item "build" -Recurse -Force }
if (Test-Path "release") { Remove-Item "release" -Recurse -Force }
Write-Host "Previous build removed." -ForegroundColor Green

# Run PyInstaller with the spec file
Write-Host "`n[1/3] Running PyInstaller..." -ForegroundColor Yellow
pyinstaller --clean --noconfirm Fallout4Toolbox.spec --log-level INFO
$pyinstallerTime = $Stopwatch.Elapsed.ToString('hh\:mm\:ss')
Write-Host "PyInstaller completed in $pyinstallerTime" -ForegroundColor Green

# Determine output app folder
$APP_DIR = Join-Path "dist" "Fallout4Toolbox"

# Copy resources
Write-Host "`n[2/3] Copying resources..." -ForegroundColor Yellow
$RESOURCE_TARGET_DIR = Join-Path $APP_DIR "resource"
if (-Not (Test-Path $RESOURCE_TARGET_DIR)) { New-Item -ItemType Directory -Path $RESOURCE_TARGET_DIR | Out-Null }
Copy-Item -Path "resource\*" -Destination $RESOURCE_TARGET_DIR -Recurse -Force
Write-Host "Resources copied." -ForegroundColor Green

# --- NEW: Create ChaiNNer info folder ---
Write-Host "Adding ChaiNNer readme..." -ForegroundColor Yellow
$CHAINNER_DIR = Join-Path $APP_DIR "ChaiNNer"
if (-Not (Test-Path $CHAINNER_DIR)) { New-Item -ItemType Directory -Path $CHAINNER_DIR | Out-Null }
Copy-Item -Path "ChaiNNer\readme.md" -Destination $CHAINNER_DIR -Force
Write-Host "ChaiNNer readme added." -ForegroundColor Green

# Trim unneeded Qt junk
$APP = $APP_DIR
Remove-Item "$APP\PySide6\Qt6WebEngine*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item "$APP\PySide6\resources\qtwebengine*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item "$APP\PySide6\Qt6Multimedia*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item "$APP\PySide6\plugins\mediaservice" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item "$APP\PySide6\plugins\audio" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item "$APP\PySide6\plugins\bearer" -Force -Recurse -ErrorAction SilentlyContinue
Get-ChildItem $APP -Recurse -Include *.pdb,*.debug | Remove-Item -Force

# Create release zip
Write-Host "`n[3/3] Creating release zip..." -ForegroundColor Yellow
$RELEASE_DIR = "release"
$ZIP_NAME = "Fallout4Toolbox_v1.0.8.zip"
if (-Not (Test-Path $RELEASE_DIR)) { New-Item -ItemType Directory -Path $RELEASE_DIR | Out-Null }

$ZIP_PATH = Join-Path $RELEASE_DIR $ZIP_NAME
if (Test-Path $ZIP_PATH) { Remove-Item $ZIP_PATH -Force }

Compress-Archive -Path "$APP_DIR\*" -DestinationPath $ZIP_PATH
$compressionTime = $Stopwatch.Elapsed.ToString('hh\:mm\:ss')

# Final summary
$Stopwatch.Stop()
Write-Host "`nBuild completed successfully!" -ForegroundColor Green
Write-Host "`nTime Summary:" -ForegroundColor Cyan
Write-Host "- PyInstaller: $pyinstallerTime"
Write-Host "- Compression: $compressionTime"
Write-Host "`nTotal Time: $($Stopwatch.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
