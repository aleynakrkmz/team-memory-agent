# Çalışma Düzeni

**Tek kural: kimse doğrudan `main`'e kod göndermez.** Herkes kendi dalında (branch) çalışır, işi bitince GitHub'da Pull Request (PR) açar, bir takım arkadaşı onaylar, sonra `main`'e birleştirilir. `main` dalı GitHub'da korunuyor; doğrudan push denersen reddedilir.

Komutlar Mac Terminal'de ve Windows PowerShell'de aynıdır. Hepsini proje klasöründeyken çalıştır (`cd team-memory-agent`).

---

## Her yeni iş için döngü

### 1. Güncel kodu al
```bash
git checkout main
git pull
uv sync
```
`uv sync`, başkası yeni paket eklediyse onu da kurar.

### 2. Kendi dalını aç
Dal adı: `isim/kısa-açıklama` (küçük harf, Türkçe karakter ve boşluk yok).
```bash
git checkout -b aleyna/sqlite-schema
```
Örnekler: `gozde/scorer`, `bahadir/agent-loop`, `aleyna/key-check`

### 3. Kodunu yaz, sık sık kaydet
```bash
git status
git add .
git commit -m "SQLite tablolarını ve append-only trigger'ları ekle"
```
- `git status` neyi değiştirdiğini gösterir; `.env` veya büyük veri dosyaları listede görünmemeli.
- Commit mesajı neyi yaptığını kısaca anlatsın.
- Bir işi birden fazla commit'e bölebilirsin.

### 4. Testleri çalıştır
```bash
uv run pytest
```
Kırmızı (failed) varsa PR açmadan önce düzelt.

### 5. Dalını GitHub'a gönder
İlk gönderişte:
```bash
git push -u origin aleyna/sqlite-schema
```
Aynı dalda sonraki gönderişlerde sadece:
```bash
git push
```

### 6. Pull Request aç
1. github.com/aleynakrkmz/team-memory-agent sayfasını aç. Üstte sarı bir kutuda **Compare & pull request** butonu çıkar, ona bas.
2. Başlığa ne yaptığını, açıklamaya kısaca neyi neden değiştirdiğini yaz.
3. Sağda **Reviewers** kısmından bir takım arkadaşını seç.
4. **Create pull request**'e bas.

### 7. İnceleme ve birleştirme
- Takım arkadaşın PR'da **Files changed** sekmesinden koda bakar, yorum yazar, uygunsa **Review changes → Approve** der.
- Düzeltme istenirse: aynı dalda düzelt, `git add .`, `git commit -m "..."`, `git push`. PR kendiliğinden güncellenir.
- Onay gelince PR sayfasında **Merge pull request → Confirm merge**. Sonra **Delete branch**.

### 8. Yerelde temizlik
```bash
git checkout main
git pull
git branch -d aleyna/sqlite-schema
```
Sonraki iş için 1. adıma dön.

---

## Kim nerede çalışır

Workstream'ler ayrı klasörlerde olduğu için çakışma az olur:

| Workstream | Lider | Ağırlıklı klasörler |
|-----------|-------|---------------------|
| 1. Veri, ses, depolama | Aleyna | `datagen/`, `app/ingest/`, `app/store/` |
| 2. Ajan ve araçlar | Bahadır | `app/extract/`, `app/match/`, `app/route/`, `app/agent/`, `prompts/` |
| 3. Değerlendirme, onay, güvenlik | Gözde | `eval/`, `app/validate/`, `app/review/`, `app/commit/` |

Başkasının klasöründe değişiklik yapman gerekirse PR açıklamasında belirt ve o kişiyi reviewer seç.

## Yapma listesi

- `.env` dosyasını commit'leme (zaten `.gitignore`'da).
- `data/` ve `runs/` içindeki veritabanlarını ve çıktıları commit'leme (zaten `.gitignore`'da).
- Paket kurmak için `pip install` kullanma; `uv add <paket>` kullan ve `pyproject.toml` ile `uv.lock`'u commit'le.
- Held-out test verisini Gözde dışında kimse açmaz (bkz. tasarım dokümanı, Bölüm 6.1).

## Sık karşılaşılan durumlar

**Yanlışlıkla `main` üzerinde değişiklik yaptım, henüz commit'lemedim:**
```bash
git checkout -b aleyna/yeni-is
```
Değişikliklerin yeni dala taşınır; normal şekilde devam et.

**`git pull` "conflict" (çakışma) dedi:**
Çakışan dosyayı aç; `<<<<<<<`, `=======`, `>>>>>>>` işaretleri arasında iki versiyon görürsün. Doğru olanı bırak, işaretleri sil, sonra:
```bash
git add .
git commit -m "Çakışmayı çöz"
```
Emin değilsen değişiklik yapmadan gruba yaz.

**PR'ım `main`'in gerisinde kaldı:**
```bash
git checkout main
git pull
git checkout aleyna/sqlite-schema
git merge main
git push
```
