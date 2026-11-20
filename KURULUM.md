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
**Bu adım neden gerekli?**
Projeyi çalıştırmak için Python 3.12 ve bazı paketler (`ollama`, `pydantic`, `rank-bm25`, `pytest` …) gerekiyor. Bunları elle kurarsak herkesin bilgisayarında farklı sürümler olur ve "bende çalışıyor, sende çalışmıyor" sorunları başlar. **uv** bu işi bizim yerimize yapan bir araç:
- Projenin istediği Python sürümünü (3.12) **kendisi indirir**. Bilgisayarında başka bir Python varsa ona dokunmaz, onu bozmaz.
- Paketleri repodaki `uv.lock` dosyasına göre **herkese birebir aynı sürümle** kurar.
- Paketleri bilgisayarın geneline değil, sadece proje klasöründeki `.venv` klasörüne kurar.

Yani uv'yi bir kez kurarsın; sonra her şey 4. adımdaki tek bir `uv sync` komutuyla olur.

**Ne yapacaksın?**
Aşağıdaki komut uv'nin resmi sitesinden (astral.sh) kurulum betiğini indirip çalıştırır. Yönetici şifresi istemez, sadece senin kullanıcı klasörüne kurulur:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Birkaç saniye içinde `everything's installed!` gibi bir mesaj görürsün.

Sonra **Terminal'i tamamen kapatıp yeniden aç** (yeni kurulan `uv` komutunu Terminal ancak yeniden açılınca tanır) ve kontrol et:
```bash
uv --version
```
`uv 0.x.x` gibi bir sürüm numarası görüyorsan tamam, 4. adıma geç.

> `uv: command not found` görürsen: Terminal'i kapatıp açmayı unutmuş olabilirsin. Hâlâ olmuyorsa şu komutu çalıştırıp tekrar dene: `source $HOME/.local/bin/env`

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
**Bu adım neden gerekli?**
Projeyi çalıştırmak için Python 3.12 ve bazı paketler (`ollama`, `pydantic`, `rank-bm25`, `pytest` …) gerekiyor. Bunları elle kurarsak herkesin bilgisayarında farklı sürümler olur ve "bende çalışıyor, sende çalışmıyor" sorunları başlar. **uv** bu işi bizim yerimize yapan bir araç:
- Projenin istediği Python sürümünü (3.12) **kendisi indirir**. Bilgisayarında başka bir Python varsa ona dokunmaz, onu bozmaz.
- Paketleri repodaki `uv.lock` dosyasına göre **herkese birebir aynı sürümle** kurar.
- Paketleri bilgisayarın geneline değil, sadece proje klasöründeki `.venv` klasörüne kurar.

Yani uv'yi bir kez kurarsın; sonra her şey 4. adımdaki tek bir `uv sync` komutuyla olur.

**Ne yapacaksın?**
Aşağıdaki komut uv'nin resmi sitesinden (astral.sh) kurulum betiğini indirip çalıştırır. Yönetici yetkisi istemez, sadece senin kullanıcı klasörüne kurulur. (`-ExecutionPolicy ByPass` kısmı, Windows'un bu tek kurulum betiğini çalıştırmasına izin verir; sistem ayarını kalıcı olarak değiştirmez.)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Birkaç saniye içinde `everything's installed!` gibi bir mesaj görürsün.

Sonra **PowerShell'i tamamen kapatıp yeniden aç** (yeni kurulan `uv` komutunu PowerShell ancak yeniden açılınca tanır) ve kontrol et:
```powershell
uv --version
```
`uv 0.x.x` gibi bir sürüm numarası görüyorsan tamam, 4. adıma geç.

> `uv is not recognized` görürsen: PowerShell'i kapatıp açmayı unutmuş olabilirsin. Hâlâ olmuyorsa bilgisayarı yeniden başlatıp tekrar dene.

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
