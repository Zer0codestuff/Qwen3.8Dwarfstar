# DwarfStar Qwen

Inference locale di **Qwen 3.8 27B** ottimizzata per Mac Apple Silicon con
**16 GB di memoria unificata**. Il percorso principale usa MLX e MLX-VLM con
un checkpoint text-only 3-bit; il precedente engine C/Metal derivato da
[`antirez/ds4`](https://github.com/antirez/ds4) resta nel repository come
implementazione legacy e riferimento sperimentale.

> Stato: beta. Il modello entra in 16 GB solo con margini stretti. Il profilo
> low-memory, il limite di contesto e la singola sessione sono requisiti di
> affidabilità, non semplici impostazioni conservative.

[English quick start](#english-quick-start) ·
[Audit e benchmark M4 16 GB](AUDIT_QWEN38.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

## Avvio rapido

Requisiti: Mac Apple Silicon, macOS recente, Python 3.10 o successivo, circa
13 GB liberi su disco per i pesi più lo spazio della cache Python.

```sh
make qwen-setup
make qwen-download
make qwen-doctor
make qwen-bench
make qwen-chat
```

`qwen-setup` crea `.venv` e installa versioni bloccate delle dipendenze dirette.
`qwen-download` scarica il target e la testa MTP alle revisioni immutabili
elencate sotto. I file vengono salvati nella normale cache Hugging Face e i
download interrotti possono essere ripresi.

Per installare solo il target seriale e risparmiare circa 213 MB di cache
(186 MB sono pesi):

```sh
make qwen-download QWEN_DOWNLOAD_ARGS=--no-mtp
```

Mostra tutti i comandi Qwen:

```sh
make qwen-help
```

## Modelli e versioni bloccate

| Componente | Identificatore | Revisione/versione | Pesi |
| --- | --- | --- | ---: |
| Target | `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | `c98bba5926f51fec1c8d8737e577221673f524d7` | 11,771 GB |
| Testa MTP | `lukaskremla/Qwen3.8-27B-MTP-3bit-MLX` | `9d061a0661258e75b401a11ac9fa22fc648e039d` | 0,186 GB |
| MLX | Apple MLX | `0.32.1` | — |
| MLX-VLM | Blaizzy MLX-VLM | `0.6.14` | — |
| Hugging Face Hub | `huggingface-hub` | `1.27.0` | — |
| Transformers / Tokenizers | Hugging Face | `5.15.0` / `0.22.2` | — |

Le revisioni predefinite vengono risolte in directory snapshot immutabili:
un aggiornamento futuro del ramo `main` su Hugging Face non cambia
silenziosamente il runtime installato. Modelli locali o repository scelti con
`--model` e `--mtp-model` restano responsabilità dell'utente.

Il target è **solo testo**: la vision tower è stata rimossa per farlo rientrare
nel Mac da 16 GB. Non sono supportati input immagine, audio o video.

## CLI

I comandi principali sono sessioni seriali con reasoning e riuso esatto della
KV cache tra i turni. Senza argomenti parte la chat; un testo semplice è una
domanda one-shot:

```sh
./dwarfstar                                  # chat con reasoning xhigh
./dwarfstar "Perché il cielo è blu?"         # domanda one-shot (ask)
./dwarfstar chat --profile balanced          # profilo intermedio
./dwarfstar ask --show-thinking "2^10 - 24?" # mostra anche il pensiero
```

I profili raccolgono le configurazioni validate sul Mac M4 16 GB:

| Profilo | Contesto | KV cache | Thinking | Budget pensiero |
| --- | ---: | --- | --- | ---: |
| `deep` (default) | 8.192 | quantizzata 4 bit | xhigh | 3.072 |
| `balanced` | 4.096 | quantizzata 8 bit | medium | 1.536 |
| `quick` | 1.024 | bf16 | disabilitato | — |

La quantizzazione della KV cache riguarda solo i 16 layer full-attention su
64 (il resto è stato ricorrente GatedDeltaNet): è ciò che rende possibile il
contesto lungo, perché la KV bf16 va in OOM già nel prefill di un prompt da
3.300 token. Ogni valore è sovrascrivibile (`--ctx-size`, `--kv-bits`,
`--thinking-budget`, `--reasoning-effort`, `--answer-reserve`, campionamento).
Il budget di risposta (`answer_reserve`) garantisce che il pensiero non
consumi l'intero budget di decode prima della risposta visibile.

Nella chat sono disponibili `/clear`, `/stats`, `/effort`, `/thinking`,
`/help`, `/exit`. Il wrapper conta i token del transcript a ogni turno e
rifiuta il turno prima di superare il limite KV.

Generazione one-shot tramite il CLI upstream (percorso legacy, utile per gli
override diagnostici):

```sh
./dwarfstar generate --no-mtp \
  --system "Rispondi in modo preciso e conciso." \
  "Spiega la differenza tra concorrenza e parallelismo."
```

`--mtp` è un override diagnostico. Sul Mac M4 16 GB di questa validazione è
andato in OOM anche dopo aver chiuso le applicazioni più pesanti; usare il
default `--mtp-auto`, che in questa installazione seleziona il seriale.

Le modalità MTP sono:

- `--mtp-auto`: usa la raccomandazione locale salvata dal benchmark; è il
  comportamento predefinito e resta seriale se non esiste un profilo valido;
- `--no-mtp`: decode seriale, percorso più affidabile;
- `--mtp`: forza la decodifica speculativa;
- `--mtp-block-size N`: blocco totale verificato, da 2 a 8; MLX-VLM propone
  `N-1` token draft per round. Il valore 1 non è una generazione MTP valida nel
  runtime bloccato ed è rifiutato.

I target Make accettano opzioni aggiuntive senza modificare il Makefile:

```sh
make qwen-chat QWEN_CHAT_ARGS='--profile balanced'
make qwen-bench QWEN_BENCH_ARGS='--mode both --max-tokens 64 --repeats 2'
```

## Server OpenAI-compatible

Avvia un server locale, con una sola sequenza attiva:

```sh
make qwen-server
```

Equivalente diretto:

```sh
./dwarfstar serve \
  --host 127.0.0.1 \
  --port 8080 \
  --max-sequences 1 \
  --mtp-auto
```

Esempio di richiesta:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dwarfstar-qwen",
    "messages": [{"role": "user", "content": "Scrivi una funzione quicksort in Python."}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

Non esporre il server su una rete non fidata senza `--api-key`. Su una macchina
da 16 GB non aumentare `--max-sequences`: ogni sessione aggiunge stato
ricorrente e KV cache. Il wrapper accetta soltanto il checkpoint Qwen pinned e
i suoi alias, normalizza il ruolo OpenAI `developer` in `system` e rifiuta il
caricamento remoto di modelli scelti dalla richiesta.

## Benchmark e MTP sperimentale

```sh
make qwen-bench
```

Il benchmark predefinito usa prompt complessi e misura il decode seriale
sicuro. Con `--mode both` confronta anche MTP sulla stessa macchina e registra
throughput, accettazione, memoria e parità testuale a decoding greedy;
quando il risultato è valido, salva la raccomandazione locale in
`.dwarfstar/profile.json`. Il profilo è legato alle revisioni esatte dei due
checkpoint, chip/architettura Metal, working set, versioni runtime, contesto,
prefill e block size; non viene riutilizzato con una configurazione diversa.

MTP è sperimentale perché il guadagno dipende dal prompt, dalla percentuale di
draft accettati e dalla pressione di memoria. Il target verifica sempre i token
proposti, ma il lavoro di draft e rollback può rendere MTP più lento del
seriale. Non trasferire su questo Mac i numeri della competizione MLXFast:
sono ottenuti su hardware M5 più veloce e con un runtime specializzato.

Risultato misurato il 18 agosto 2026 sul Mac M4 16 GB di destinazione:

| Prova | Risultato |
| --- | ---: |
| Decode seriale, mediana di 3 prompt complessi × 256 token | **7,35 tok/s** |
| Range decode | 6,47–7,36 tok/s |
| Prefill dei due prompt successivi al primo misurato | 39,6–42,5 tok/s |
| Picco MLX | 12,29 GB |
| Prompt 3.398 token, ctx 4K, KV 8 bit, chunk 128 | 7,58 tok/s, picco 12,49 GB |
| Prompt 7.038 token, ctx 8K, KV 4 bit, chunk 64 | 7,43 tok/s, picco 12,40 GB |
| Prompt 3.346 token con KV bf16 (chunk 256 e 128) | OOM nel prefill |
| MTP 3-bit | OOM nel warm-up, default seriale |

Il precedente percorso C/SSD misurava circa 0,04 token/s. Output grezzi,
valutazione qualitativa e problemi trovati sono documentati in
[AUDIT_QWEN38.md](AUDIT_QWEN38.md).

Per una misura solo seriale:

```sh
./dwarfstar benchmark --mode serial
```

Per forzare un confronto più lungo:

```sh
./dwarfstar benchmark --mode both --max-tokens 96 --repeats 2
```

## Limiti di memoria su 16 GB

Sul Mac M4 usato per sviluppare questo port, Metal riporta un working set
raccomandato di circa **12,71 GB**. Il target occupa 11,77 GB; con MTP i soli
pesi arrivano a circa 11,96 GB. Restano da allocare stato GatedDeltaNet, KV
cache, attivazioni, grafi e buffer temporanei.

Solo 16 dei 64 layer hanno una KV cache (circa 64 KiB/token in bf16); gli
altri usano stato ricorrente GatedDeltaNet a dimensione costante. Il collo di
bottiglia del contesto lungo non è lo storage KV ma il picco transiente del
prefill: la mitigazione misurata è quantizzare la KV cache dal token 0 e
ridurre il chunk di prefill.

Le impostazioni predefinite sono quindi:

- profilo `deep`: contesto 8.192, KV 4 bit, chunk prefill 64;
- profilo `balanced`: contesto 4.096, KV 8 bit, chunk prefill 128;
- profilo `quick` e benchmark: contesto 1.024, KV bf16, chunk prefill 256;
- una sola sequenza server;
- decode seriale finché il benchmark locale non raccomanda MTP;
- cache libera MLX limitata a 64 MB;
- fusione GDN che duplica circa 1,8 GB di pesi disattivata.

I limiti di contesto dipendono dalla quantizzazione KV e sono stati misurati
sul Mac di destinazione:

| KV cache | Limite contesto |
| --- | ---: |
| bf16 (`--kv-bits 0`) | 2.048 |
| quantizzata ≥ 6 bit | 4.096 |
| quantizzata < 6 bit | 8.192 |
| con MTP | 1.024 |

Il chunk di prefill viene ridotto automaticamente (256 fino a 2K, 128 fino a
4K, 64 oltre). L'override `DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1` è diagnostico:
può causare swap, terminazione del processo o instabilità del sistema. Il
contesto teorico da 262K del modello non è realizzabile su un Mac da 16 GB.

Per ridurre il rischio di OOM:

- chiudi applicazioni pesanti prima di caricare il modello;
- non avviare due processi del modello contemporaneamente;
- in caso di OOM ripiega su `--profile balanced` o `--profile quick`;
- esegui `make qwen-doctor` dopo setup e download;
- non abilitare `DWARFSTAR_ALLOW_FUSED_GDN=1` su 16 GB.

Variabili avanzate:

| Variabile | Uso |
| --- | --- |
| `DWARFSTAR_MODEL` | target locale o repository alternativo |
| `DWARFSTAR_MTP_MODEL` | testa MTP alternativa |
| `DWARFSTAR_MLX_CACHE_LIMIT_MB` | limite della cache libera MLX; default 64 |
| `DWARFSTAR_MLX_MEMORY_LIMIT_MB` | soglia guida MLX; default: working set Metal raccomandato |
| `DWARFSTAR_PROFILE` | percorso alternativo del profilo benchmark |
| `DWARFSTAR_RUNTIME_LOCK` | percorso alternativo del lock tra processi |
| `DWARFSTAR_ALLOW_UNSAFE_CONTEXT=1` | ignora i limiti di contesto, non raccomandato |
| `DWARFSTAR_ALLOW_UNSAFE_SEQUENCES=1` | consente più sequenze server, diagnostico |
| `DWARFSTAR_ALLOW_FUSED_GDN=1` | riabilita la fusione ad alto consumo, non usare su 16 GB |

## Test

I test rapidi non caricano il modello:

```sh
make qwen-test
```

Prima di considerare valida una configurazione reale:

```sh
make qwen-doctor
make qwen-bench
```

Valuta inoltre output italiani, codice, matematica, Unicode, tool call e
prompt lunghi. Il 3-bit è il più grande checkpoint affine uniforme validato
qui che rientra nel profilo, ma può perdere qualità rispetto a quantizzazioni
più grandi.

## Engine C/Metal legacy

I file `ds4*.c`, `ds4_metal.m`, `metal/`, `cuda/` e `rocm/` appartengono al
progetto nativo originale. Sono conservati per compatibilità, studio e futuri
port Metal diretti. Il percorso Qwen in quel codice è un riferimento CPU
sperimentale: non è il backend Qwen raccomandato su questo Mac.

I target legacy restano invariati:

```sh
make all             # build C/Metal legacy su macOS
make cpu             # build CPU diagnostica
make cuda-spark      # CUDA per DGX Spark
make cuda-generic    # CUDA generica
make strix-halo      # ROCm per Strix Halo
make test            # suite legacy
make help            # elenco completo
```

Il nuovo percorso MLX non modifica il formato GGUF né promette compatibilità
con i modelli DeepSeek/GLM supportati dal vecchio engine.

## Riconoscimenti e licenze

Il progetto deriva dall'engine [antirez/ds4](https://github.com/antirez/ds4)
e usa MLX e MLX-VLM. I criteri di benchmark prendono ispirazione dalla
[Qwen 3.8 MLXFast challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge);
decoding speculativo e rollback sono forniti da MLX-VLM.
Qwen e i checkpoint quantizzati mantengono la licenza e le condizioni del
modello upstream. Vedi [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) e
[LICENSE](LICENSE).

---

## English quick start

DwarfStar Qwen runs the **text-only Qwen 3.8 27B 3-bit MLX checkpoint** on a
16 GB Apple Silicon Mac. The MLX wrapper is the primary path. The older native
C/Metal ds4 engine remains available as a legacy, experimental reference.

```sh
make qwen-setup
make qwen-download
make qwen-doctor
make qwen-bench
make qwen-chat
```

Start the OpenAI-compatible local server with:

```sh
make qwen-server
```

Useful direct commands:

```sh
./dwarfstar "Explain lock-free queues with a small example."   # one-shot ask
./dwarfstar chat --profile balanced
./dwarfstar benchmark --mode serial
./dwarfstar serve --host 127.0.0.1 --port 8080 --max-sequences 1
```

The default `deep` profile runs xhigh thinking in an 8,192-token context with
a 4-bit quantized KV cache and a 64-token prefill chunk, the configuration
validated on the M4 16 GB machine. `balanced` uses 4,096 tokens with 8-bit KV;
`quick` disables thinking at 1,024 tokens. Context caps depend on KV
quantization (2,048 for bf16 KV, 4,096 at 8-bit, 8,192 at 4-bit, 1,024 with
MTP). MTP is experimental and stays disabled until a valid local benchmark
profile recommends it. Run only one model process and one active server
sequence.

Default model revisions are immutable:

- target: `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` at
  `c98bba5926f51fec1c8d8737e577221673f524d7`;
- MTP head: `lukaskremla/Qwen3.8-27B-MTP-3bit-MLX` at
  `9d061a0661258e75b401a11ac9fa22fc648e039d`.

This build is text-only and does not accept images, audio or video. See the
Italian sections above for memory accounting, MTP caveats, environment
variables and the legacy native build commands.
