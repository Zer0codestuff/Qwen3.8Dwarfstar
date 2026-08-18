# Audit e validazione — DwarfStar Qwen 3.8 27B

Data della prova: 18 agosto 2026. Hardware: MacBook Apple M4, 10 core GPU,
16 GiB di memoria unificata, macOS 26.5.2. Tutte le misure riportate qui sono
state eseguite sulla macchina di destinazione, non estrapolate da benchmark
pubblici.

## Esito

Il port C/Metal trovato nel repository non era un engine Qwen utilizzabile su
questo Mac: il forward CPU produceva testo plausibile, ma lo streaming dei pesi
dal disco raggiungeva circa 0,04 token/s, mentre il GGUF da 13,82 GB non entrava
nel working set Metal. Il percorso supportato è stato quindi ricostruito sopra
MLX-VLM con un checkpoint text-only affine 3-bit da 11,77 GB e un profilo
low-memory specifico per 16 GB.

Il risultato operativo è stabile in decoding seriale: mediana 7,35 token/s sui
tre prompt complessi, picco massimo 12,29 GB e server OpenAI-compatible
funzionante. MTP non è attivabile in sicurezza con questa combinazione di
hardware, pesi 3-bit e kernel correnti: due prove con block size 3 e una con il
minimo valido 2, eseguite anche dopo aver chiuso Zen, sono terminate con
`kIOGPUCommandBufferCallbackErrorOutOfMemory` nel warm-up. Il runtime lo
registra e mantiene il seriale come default.

## Problemi trovati nel port nativo

La parte matematica principale di Qwen 3.8 era più completa di quanto lasciasse
supporre il comportamento esterno: GatedDeltaNet, convoluzione, RMSNorm Q/K,
GQA, gated attention e mapping delle teste erano sostanzialmente allineati al
layout GGUF atteso. I problemi bloccanti erano soprattutto di integrazione e di
architettura runtime.

1. **Backend incompatibile con 16 GB.** Il GGUF Q3_K_M locale pesa 13,819 GB,
   contro 12,713 GB di working set Metal raccomandato. Il full offload di
   llama.cpp è andato in OOM; il percorso nativo leggeva invece circa 10–14 GB
   di layer dal disco per ogni token.
2. **Chat template Qwen non affidabile.** Alcuni ingressi venivano tokenizzati
   come testo grezzo o passavano dal renderer DeepSeek; la costruzione manuale
   dei marker ChatML non riproduceva il template ufficiale. Questo cambia in
   modo sostanziale la distribuzione vista dal modello.
3. **Tokenizer whitespace divergente.** Il port non implementava il look-ahead
   Qwen per spazi ripetuti e indentazione. Esempi golden:
   `alpha  beta` deve diventare `[6918, 220, 13053]`; una riga Python con quattro
   spazi deve conservare l'indentazione secondo il tokenizer ufficiale.
4. **Stato ricorrente non isolato per sessione.** Il runtime Qwen possedeva un
   solo stato GDN/conv/KV mutabile; sessioni interlacciate, rewind e restore
   potevano quindi contaminarsi.
5. **Contesto mascherato.** Richieste oltre 4.096 token potevano essere ridotte
   silenziosamente a 512 invece di essere rifiutate con un errore esplicito.
6. **Dispatch Metal e stop incompleti.** Qwen poteva entrare nel batch Metal
   generico senza un grafo Qwen inizializzato; inoltre occorre gestire entrambi
   gli stop ufficiali, `248046` (`<|im_end|>`) e `248044`
   (`<|endoftext|>`).
7. **Persistenza dichiarata più ampia della realtà.** Lo snapshot della
   sessione non copriva correttamente tutto lo stato ricorrente Qwen.

Questi difetti non sono stati nascosti dietro piccoli fix locali: il C nativo è
rimasto esplicitamente un backend legacy sperimentale. Correggerlo fino a un
Metal full-resident richiederebbe anche un formato di quantizzazione più piccolo
e kernel Qwen dedicati; il solo refactoring delle API non risolverebbe il limite
fisico dei pesi.

## Soluzione implementata

- checkpoint target
  `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` bloccato alla revisione
  `c98bba5926f51fec1c8d8737e577221673f524d7`;
- testa MTP opzionale bloccata alla revisione
  `9d061a0661258e75b401a11ac9fa22fc648e039d`;
- versioni dirette bloccate: MLX 0.32.1, MLX-VLM 0.6.14,
  Transformers 5.15.0 e Tokenizers 0.22.2;
- template chat e tokenizer del checkpoint, senza BOS spurio e con i due stop
  token ufficiali;
- contesto predefinito 1.024, chunk prefill 256 e una sola sequenza server;
- cache libera MLX limitata a 64 MB;
- soglia guida dell'allocatore MLX ridotta dal default upstream, superiore al
  working set raccomandato, al valore raccomandato da Metal; non è un hard cap;
- disabilitazione della fusione GDN che, sul target 3-bit, conserverebbe una
  copia concatenata aggiuntiva di circa 1,77 GB;
- download e risoluzione a snapshot immutabile con verifica della dimensione
  totale dei file dei pesi;
- benchmark A/B che abilita MTP solo con parità del testo a decoding greedy,
  almeno +3% di velocità e
  memoria entro il limite; gli errori Metal diventano un risultato registrato,
  non un crash;
- terminazione sicura del worker dopo il flush. Con il 27B quasi al limite,
  la finalizzazione parziale degli oggetti Metal provocava un segfault 139 dopo
  una generazione già completata; le risorse vengono ora restituite
  atomicamente dal sistema operativo;
- server locale OpenAI-compatible con alias stabili `dwarfstar-qwen` e
  `qwen3.8-27b`, allowlist del solo target pinned e normalizzazione del ruolo
  `developer` in `system`;
- lock di processo: chat, server e benchmark non possono caricare due copie del
  27B contemporaneamente;
- chat interattiva limitata al percorso seriale/no-thinking, con conteggio duro
  del transcript prima del limite KV. Reasoning multi-turn e MTP vengono
  rifiutati perché il ramo chat upstream non li conserva/inoltra correttamente.

## Memoria misurata

| Voce | Valore |
| --- | ---: |
| Memoria unificata fisica | 17,18 GB / 16 GiB |
| Working set Metal raccomandato | 12,713 GB |
| Pesi target 3-bit | 11,771 GB |
| Pesi MTP aggiuntivi | 0,186 GB |
| Picco seriale, prompt complessi | 12,248–12,294 GB |
| Margine sul picco peggiore | circa 419 MB |
| MTP, block size 2 e 3 | OOM nel warm-up, anche con Zen chiuso |

Il margine è sufficiente per una sola sessione breve, ma non rende il modello
indifferente alle altre applicazioni: RAM e banda sono unificate. Chiudere app
pesanti riduce il rischio di pressione/swap; non ha però reso MTP eseguibile.

## Prestazioni

| Percorso | Carico | Decode | Prefill | Picco | Esito |
| --- | --- | ---: | ---: | ---: | --- |
| C nativo iniziale | 32 token, prompt semplice | 0,04 tok/s | 0,04 tok/s | RSS 4,18 GB | 920,91 s |
| C nativo iniziale | aritmetica, 16 token | 0,04 tok/s | 0,04 tok/s | RSS 4,04 GB | 1.208,27 s, risposta tronca |
| C nativo iniziale | codice, 24 token | 0,04 tok/s | 0,04 tok/s | RSS 3,97 GB | 1.370,21 s, risposta tronca |
| MLX seriale finale | 3 prompt × 256 token | **7,35 tok/s mediana** | 7,3–42,5 tok/s | 12,29 GB | stabile |
| MLX seriale, runtime finale | 3 prompt × 32 token | 5,31 tok/s mediana | 35,2–41,5 tok/s | 12,29 GB | stabile |
| Server caldo | richiesta breve | 5,1 tok/s | 23,5 tok/s | 12,09 GB | HTTP 200 |
| MTP 3-bit, block size 2/3 | warm-up 4 token | — | — | oltre budget | OOM |

Il decode complesso seriale finale varia fra 6,47 e 7,36 token/s: circa **184×** il
vecchio percorso da 0,04 token/s. Il primo prefill dopo il caricamento è più
lento per compilazione/warm-up; i successivi sono sensibilmente più veloci. La
misura finale è stata eseguita dopo la chiusura di Zen e conferma che le app
aperte influenzano la banda/pressione della memoria unificata.
I dati grezzi versionabili sono `benchmark-results/qwen38-m4-complex.json`,
`qwen38-m4-final-smoke.json` e `qwen38-m4-mtp-block2.json`.

## Qualità sui prompt complessi

La quantizzazione 3-bit è il compromesso che rende possibile il caricamento,
non una garanzia di qualità equivalente al modello a precisione maggiore.

- **Matematica:** il modello ha impostato il metodo, ma in modalità no-thinking
  ha scritto un valore errato per `C(30,6)` ed è arrivato al limite di output.
  L'oracolo è
  `(C(30,6)-C(18,6)-C(20,6)-C(22,6)+C(8,6)+C(10,6)+C(12,6))/C(30,6)` =
  **18520/23751**.
- **Python async:** individua correttamente la race check-then-act, ma la
  soluzione con lock per chiave rimane incompleta rispetto a cancellazione,
  condivisione del task in-flight e pulizia dei lock.
- **Italiano tecnico:** la prosa è fluida, ma attribuisce una cache KV troppo
  grande e tratta l'architettura come Transformer puro, omettendo lo stato
  ricorrente GDN.
- **Reasoning medium (prova manuale):** costruisce correttamente
  l'inclusione-esclusione e si autocorregge sulla mappatura dei colori, ma
  consuma tutti i 512 token nel blocco di pensiero senza emettere una risposta
  finale.

Conclusione qualitativa: buono per chat, sintesi e coding leggero con verifica
umana; non va usato come fonte non verificata per calcoli combinatori, stime di
memoria o concorrenza delicata. Per questi casi servono un budget più ampio,
prompt più vincolanti e una verifica esterna, oppure un checkpoint meno
aggressivamente quantizzato su hardware con più memoria.

## Verifiche eseguite

- `dwarfstar doctor`: hardware, versioni, Metal e dimensione dei due checkpoint OK;
- 27/27 test Qwen, inclusi golden ChatML, token speciali, whitespace e
  indentazione Python;
- compilazione Python e `pip check` senza errori;
- build e suite legacy C/Metal passate; i soli test che richiedono il vecchio
  `ds4flash.gguf` vengono saltati esplicitamente quando il file non è presente;
- generazione reale one-shot, incluso arresto pulito dopo la correzione del
  teardown;
- benchmark complesso seriale, benchmark breve sul runtime finale e tre
  tentativi MTP (block size 2 e 3); MTP-only restituisce correttamente exit 4;
- smoke seriale sul runtime finale con la nuova soglia MLX e contesto 2K:
  `DUEMILA OK`, exit 0, 6,86 token/s, 12,07 GB MLX;
- server reale: `GET /v1/models`, due `POST /v1/chat/completions` sequenziali,
  alias differenti, risposte `ALFA` e `BETA`, HTTP 200, nessuna contaminazione
  fra richieste e shutdown pulito; ruolo `developer` normalizzato con risposta
  `GAMMA`/`OK`; modello remoto arbitrario e `max_tokens: 0` rifiutati con
  HTTP 400;
- chat interattiva reale: risposta `CHAT OK`, limite KV inoltrato e `Ctrl-D`
  pulito con exit 0; secondo processo concorrente rifiutato con exit 2.

## Addendum — contesto lungo e KV quantizzata (18 agosto 2026)

La versione 0.3.0 ha introdotto le sessioni con reasoning (`ask`/`chat`).
Le prove di prefill lungo eseguite sulla stessa macchina hanno mostrato che il
limite reale non è lo storage della KV cache (solo 16 layer su 64 ne hanno
una, circa 64 KiB/token in bf16) ma il picco transiente del prefill:

| Prova (prompt reale, retrieval esatto verificato) | Esito |
| --- | --- |
| 3.346 token, ctx 4096, KV bf16, chunk 256 | OOM nel prefill |
| 3.346 token, ctx 4096, KV bf16, chunk 128 | OOM nel prefill |
| 3.398 token, ctx 4096, KV 8 bit, chunk 128 | OK: 7,58 tok/s, picco 12,49 GB |
| 6.986 token, ctx 8192, KV 8 bit, chunk 64 | OOM nel prefill |
| 7.038 token, ctx 8192, KV 4 bit, chunk 64 | OK: 7,43 tok/s, picco 12,40 GB |
| Chat 2 turni, default deep (8192/kv4/chunk 64) | OK: 175 token riusati dalla cache, 7,28 tok/s, picco 12,08 GB |

Politica adottata: limiti di contesto legati alla quantizzazione KV (bf16 →
2.048; ≥ 6 bit → 4.096; < 6 bit → 8.192; MTP → 1.024), chunk di prefill
ridotto automaticamente (256/128/64) e `--quantized-kv-start 0` obbligatorio
quando la KV è quantizzata (il default upstream, token 5000, la renderebbe un
no-op entro questi limiti). I profili `deep` (8192/kv4, thinking xhigh) e
`balanced` (4096/kv8, thinking medium) usano solo configurazioni misurate. Il
checkpoint 2-bit è stato valutato e scartato: libererebbe circa 3,5 GB ma
degrada proprio la qualità del ragionamento che motiva questo runtime.

## Limiti residui e uso raccomandato

1. Il target è solo testo; immagini, audio e video non sono supportati.
2. Usare i profili validati: `deep` 8K con KV 4 bit, `balanced` 4K con KV
   8 bit, `quick` 1K bf16. Senza KV quantizzata il limite è 2K. Il limite
   teorico 262K del modello non è praticabile su 16 GB.
3. Lasciare MTP in `auto`/seriale. Anche il minimo block size valido (2, cioè un
   token draft) va in OOM. Un kernel q3 dedicato o un checkpoint più piccolo
   resta la strada più promettente.
4. Non avviare più processi del 27B contemporaneamente e chiudere applicazioni
   pesanti quando serve il massimo margine.
5. Verificare sempre le risposte tecniche importanti: gli errori osservati sono
   del modello/quantizzazione, non del solo renderer.

Comandi essenziali:

```sh
make qwen-setup
make qwen-download
make qwen-doctor
./dwarfstar chat --no-mtp
./dwarfstar benchmark --mode serial --max-tokens 256
./dwarfstar serve --no-mtp --host 127.0.0.1 --port 8080 --max-sequences 1
```

## Fonti primarie

- modello ufficiale: <https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0>;
- target 3-bit esatto: <https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly/tree/c98bba5926f51fec1c8d8737e577221673f524d7>;
- MTP 3-bit esatto: <https://huggingface.co/lukaskremla/Qwen3.8-27B-MTP-3bit-MLX/tree/9d061a0661258e75b401a11ac9fa22fc648e039d>;
- MLX-VLM 0.6.14: <https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.14>;
- challenge e snapshot di implementazione studiato:
  <https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/tree/8dabcfb75c19dad6cbf5cc7cf9f26e1bd440a0dd>;
- crown live osservata alle 09:17 UTC del 18 agosto 2026:
  <https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/tree/d56b4a0eb4e52f2fb92540ba5c6ed764176e8d33>;
- leaderboard MLXFast live (può cambiare):
  <https://www.yukon.org/mlxfast>;
- implementazione Qwen di confronto in llama.cpp:
  <https://github.com/ggml-org/llama.cpp/blob/82dbc4f017a7b005f993ac2e7af9c048ad686c04/src/models/qwen35.cpp>;
- base legacy antirez/ds4:
  <https://github.com/antirez/ds4/tree/84cc882352757baf628a1776badf7cc54d584e28>.

Alle 09:17 UTC del 18 agosto 2026 la crown live mostrava 84,2 token/s median su
otto sequenze da 512 token. Il dato è ottenuto su M5 Max 128 GB con kernel e
target 4-bit specializzati e la classifica cambia nel tempo: è un confronto di
ricerca per scheduling e fusione, non un obiettivo trasferibile al M4 16 GB.
