#!/bin/bash

# 🏗 Ustal ścieżkę do folderu projektu niezależnie od lokalizacji skryptu
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$PROJECT_DIR/models/deepface_weights"
DEEPFACE_SYMLINK="$HOME/.deepface/weights"

echo "📦 Projekt: Face Recognition"
echo "🎯 Cel: Skopiować modele DeepFace i utworzyć symlink"

# 🗂 Utwórz folder docelowy na modele, jeśli nie istnieje
mkdir -p "$TARGET_DIR"
mkdir -p "$HOME/.deepface"

# 🔁 Przenieś istniejące modele, jeśli są
if [ -d "$HOME/.deepface/weights" ] && [ ! -L "$HOME/.deepface/weights" ]; then
    echo "🟡 Przenoszę istniejące modele do $TARGET_DIR..."
    mv "$HOME/.deepface/weights"/* "$TARGET_DIR" 2>/dev/null
    rmdir "$HOME/.deepface/weights" 2>/dev/null
fi

# 🧹 Usuń istniejący symlink, jeśli jest zły
if [ -L "$DEEPFACE_SYMLINK" ] && [ ! -e "$DEEPFACE_SYMLINK" ]; then
    echo "🧹 Usuwam stary nieprawidłowy symlink..."
    rm "$DEEPFACE_SYMLINK"
fi

# 🔗 Tworzenie dowiązania symbolicznego
if [ ! -L "$DEEPFACE_SYMLINK" ]; then
    ln -s "$TARGET_DIR" "$DEEPFACE_SYMLINK"
    echo "✅ Symlink utworzony: $DEEPFACE_SYMLINK -> $TARGET_DIR"
else
    echo "✅ Symlink już istnieje: $DEEPFACE_SYMLINK"
fi

echo "🎉 Gotowe! DeepFace używa teraz modeli z folderu projektu."
