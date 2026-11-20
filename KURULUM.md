# Kurulum Rehberi

Bu rehberi bir kez, baştan sona takip et. Sonunda `uv run pytest` komutu yeşil sonuç veriyorsa ortamın hazır demektir.

Kendi işletim sistemine ait bölümü takip et: **[Mac](#mac)** veya **[Windows](#windows)**.

> Komutları **Terminal**'e (Mac) veya **PowerShell**'e (Windows) yapıştırıp Enter'a bas. Her komutun bitmesini bekle.

---

## Mac

### 1. Git
Terminal'i aç (Spotlight → "Terminal") ve yaz:
```bash
git --version
```
Sürüm numarası çıkıyorsa Git kurulu, 2. adıma geç. Bir pencere açılıp "command line developer tools" kurmayı önerirse **Install**'a bas, bitince komutu tekrar dene.

### 2. GitHub'a SSH anahtarı ekle (bir kez)
Bu adım, her seferinde şifre girmeden kodu indirip gönderebilmen içindir.
```bash
ssh-keygen -t ed25519 -C "github"
```
Sorulan her şeye Enter'a basarak geç. Sonra anahtarı kopyala:
```bash
pbcopy < ~/.ssh/id_ed25519.pub
```
GitHub'da sağ üstte profil fotoğrafı → **Settings** → **SSH and GPG keys** → **New SSH key** → *Key* kutusuna yapıştır → **Add SSH key**.

Kontrol et:
```bash
ssh -T git@github.com
```
İlk seferde "Are you sure you want to continue connecting?" diye sorarsa `yes` yaz. `Hi <kullanıcı-adın>! You've successfully authenticated` görürsen tamam.

### 3. uv (Python ve paket yöneticisi)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Bittikten sonra **Terminal'i kapatıp yeniden aç**, sonra kontrol et:
```bash
uv --version
```

### 4. Projeyi indir ve kur
```bash
cd ~/Desktop
git clone git@github.com:aleynakrkmz/team-memory-agent.git
cd team-memory-agent
uv sync
cp .env.example .env
```
`uv sync` projenin kullandığı Python 3.12'yi ve tüm paketleri otomatik indirir. Bilgisayarındaki başka Python sürümlerine dokunmaz.

### 5. Ollama
[ollama.com/download](https://ollama.com/download) adresinden Mac sürümünü indir, `.zip`'i aç, Ollama'yı **Applications** klasörüne sürükle ve bir kez çalıştır. Modellerin listesi ekip olarak seçildiğinde `.env.example` içinde olacak; o zaman her birini şöyle indireceğiz:
```bash
ollama pull <model-adı>
```

### 6. Kontrol
```bash
uv run pytest
```
Bkz. [Sonucu nasıl okurum?](#sonucu-nasıl-okurum)

---

## Windows

### 1. Git
[git-scm.com/download/win](https://git-scm.com/download/win) adresinden Git'i indir ve kur. Kurulumdaki bütün seçenekleri varsayılan bırakıp **Next** ile geçebilirsin.

Sonra **PowerShell**'i aç (Başlat menüsü → "PowerShell") ve kontrol et:
```powershell
git --version
```

### 2. GitHub'a SSH anahtarı ekle (bir kez)
```powershell
ssh-keygen -t ed25519 -C "github"
```
Sorulan her şeye Enter'a basarak geç. Sonra anahtarı kopyala:
```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard
```
GitHub'da sağ üstte profil fotoğrafı → **Settings** → **SSH and GPG keys** → **New SSH key** → *Key* kutusuna yapıştır → **Add SSH key**.

Kontrol et:
```powershell
ssh -T git@github.com
```
İlk seferde soru sorarsa `yes` yaz. `Hi <kullanıcı-adın>! You've successfully authenticated` görürsen tamam.

### 3. uv (Python ve paket yöneticisi)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Bittikten sonra **PowerShell'i kapatıp yeniden aç**, sonra kontrol et:
```powershell
uv --version
```

### 4. Projeyi indir ve kur
```powershell
cd $env:USERPROFILE\Desktop
git clone git@github.com:aleynakrkmz/team-memory-agent.git
cd team-memory-agent
uv sync
Copy-Item .env.example .env
```

### 5. Ollama
[ollama.com/download](https://ollama.com/download) adresinden Windows sürümünü indir ve kur. Model listesi belli olunca her birini şöyle indireceğiz:
```powershell
ollama pull <model-adı>
```

### 6. Kontrol
```powershell
uv run pytest
```

---

## Sonucu nasıl okurum?

Örnek başarılı çıktı:
```
23 passed, 1 skipped
```
- **passed** → her şey yolunda.
- **1 skipped** → Ollama henüz çalışmıyor demektir. Ollama'yı kurup açtıktan sonra bu test de *passed* olur. Kurulumu tamamlamana engel değil.
- **failed** veya kırmızı yazı → hata mesajının ekran görüntüsünü gruba at.

## Sık karşılaşılan sorunlar

| Sorun | Çözüm |
|-------|-------|
| `uv: command not found` / `uv is not recognized` | Terminal'i/PowerShell'i kapatıp yeniden aç. |
| `Permission denied (publickey)` | 2. adımdaki SSH anahtarı GitHub'a eklenmemiş. Tekrar kontrol et. |
| `Repository not found` | Repo davetini kabul etmemişsin: e-postandaki daveti veya github.com/aleynakrkmz/team-memory-agent/invitations sayfasını aç. |
| Python sürümü hatası | Komutları `python ...` değil, `uv run ...` ile çalıştır. |

## Günlük kullanım

- Kod çalıştırmak: `uv run python <dosya>.py`
- Testleri çalıştırmak: `uv run pytest`
- Biri yeni paket eklediyse: `git pull` sonra `uv sync`
- Yeni paket eklemek: `uv add <paket-adı>` (sonra `pyproject.toml` ve `uv.lock` dosyalarını commit'le)

Kod gönderme düzeni için: [CALISMA_DUZENI.md](CALISMA_DUZENI.md)
