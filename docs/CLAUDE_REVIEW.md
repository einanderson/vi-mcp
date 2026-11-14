# VI Bewertung und Prüfauftrag für Claude

Stand: 14. November 2026. Autor: Codex. Empfänger: Claude als unabhängiger Codeprüfer.
Ausgangsstand: `e7acd14929f69c33a5278997f43fa05f165ec2e1`. Zielstand: 0.64.0, noch nicht veröffentlicht.
Implementierung des Maschinen- und kNN-Kerns: `3cd332a4c07ad6d594e777cc48da5aedf8335a8a`.
Vollständiger Instanzvertrag mit kohärenten Lernbelegen und Checkpoints: `a96768490970b0651d7a84ab7c871d4535671938`.
Reale Trainer-Vorprüfung, vollständige Lernziele und eigenständige Evaluatoren: `01ed43b991043cff06e154594738f76476cf608a`.

## Ergebnis

vi-mcp hat jetzt eine ausführbare virtuelle Intelligenz im hier definierten technischen Sinn: Ein austauschbares LLM programmiert
mit VIL und clingo-object eine persistente Maschine mit veränderlichen Daten. Der Runtime gehören Zustand, Ausführung, Isolation,
Snapshots und Prüfnachweise. Ein unverändertes Programm liefert nach einer Änderung der Fakten ein anderes korrektes Ergebnis.
[Der Maschinenvertrag](VIRTUAL_MACHINE.md) definiert die unterstützte Semantik und Grenzen.

Der [vollständige Instanzvertrag](INSTANCE_RUNTIME.md) ergänzt jetzt den eigenen Kontext, Modell- und Decoderbindung,
MCP-Geräte, ausdrückliche Rechte und eine kontrollierte Ausführungsschleife. Ein kohärenter lokaler Checkpoint sichert diesen
Zustand zusammen mit dem neuronalen Datastore; ein Klon erhält eine eigene Änderungsebene bei gemeinsamem eingefrorenem Kern.

Zusätzlich ist [kNN-LM im Inferenzkern](KNN_LM.md) implementiert: Ein austauschbarer Speicher echter Modellzustände und Folgetokens
verändert die Tokenverteilung bei unveränderten Modellgewichten. Er läuft als eigener Transformers-Inferenzbackend unterhalb von
MCP und kann den vorhandenen Modell-Client der VI bedienen. Titans ist nicht implementiert.

Das ist ein funktionsfähiger erster Maschinenvertrag. Die Bezeichnung „vollwertige VI“ ist außerhalb eines solchen Vertrags kein
messbarer Standard. Allgemeine Intelligenz, Bewusstsein oder eine QEMU-artige Emulation neuronaler Gewichte sind damit nicht
nachgewiesen. Das Projekt virtualisiert die symbolische Arbeitsumgebung und Programmausführung eines LLM.

| Bereich | Bewertung | Entscheidender Nachweis oder Grenze |
|---|---|---|
| Persistentes Gedächtnis und Suche | Technisch weit entwickelt | Gemessene Suchqualität; Mehrkosten der hybriden Suche |
| Veränderliche virtuelle Maschine | Ausführbarer Vertrag v1 | Gleicher Programmtext, neue Fakten, neue Antwort; Restore und Fork |
| Vollständige Instanz mit Kontext und Geräten | Implementiert und funktional geprüft | Kohärenter neuronaler Checkpoint, Boot, Geräte-/Rechteprüfung, echte LLM-Schleife |
| LLM als Programmierer | In einem einfachen realen Versuch bestätigt | Zwei richtige Antworten bei zwei Datenständen; eine Aufgabenfamilie |
| kNN-LM unterhalb von MCP | Implementiert und mit vortrainiertem Modell ausgeführt | Ersetzen, Rückziehen und Snapshot ändern die Tokenvorhersage bei gleichem Gewichtshash |
| KI-Lerninhalte | Vorhanden, jetzt genauer geprüft und partitioniert | SFT, KTO, DPO und ausführbare Referenzfälle |
| Trainer und Tokenizer | Alle fünf Stufen funktional vorgeprüft | Echte TRL/PEFT-Trainer, vollständige Ziele, endlicher Loss, null Optimierungsschritte |
| Lerngewinn durch Fine-Tuning | Offen | Noch kein Training und kein Vergleich eines trainierten Adapters |
| Gesamtnutzen der Hinweise | Offen | Bisherige randomisierte Messung noch zu unsicher |
| Schutz gegen vergiftetes Wissen | Begrenzte Robustheit | Feste Regeln generalisieren schlecht; Daten werden nicht als ausführbarer Python-Code behandelt |

## Befunde am bisherigen System

Die folgenden Zahlen stammen aus der Untersuchung vom 14. November. Der Trainingsdaten-Audit verwendete eine temporäre Kopie der
Produktionsdatenbank des installierten Stands 0.63.2; die Quelle wurde ausschließlich lesend geöffnet. Die Zahlen beschreiben den
Ausgangszustand, keine erneute Produktionsmessung des neuen Codes.

- Hybride Suche: MRR 0,851 gegenüber 0,407 bei Volltext; Treffer unter den ersten fünf 0,933 gegenüber 0,590. Die gemessene Laufzeit
  lag bei etwa 1.813 gegenüber 87 ms. Das belegt Suchqualität und deren Kosten, keinen vollständigen Agentennutzen.
- Randomisierter Vorteil der Hinweisauswahl: 4,7 ± 3,1 Prozentpunkte, wobei ± einen Standardfehler bezeichnet. Dieser Zeitraum
  liefert keinen belastbaren positiven Effekt. Die Kontrollgruppe ohne Hinweise enthielt nur 21 Aktionen. Direktes Feedback
  wurde nur bei ungefähr 2 % der Hinweise gegeben.
- Von 24.469 Wissensobjekten waren 547 geprüft und 505 klassifiziert; unabhängige Goldbewertungen fehlten. Eine große Datenmenge
  allein ist daher kein Reifebeleg für den Inkubator.
- Der Export enthielt 24.989 QA-, 1.595 agentische, 143 Logik-, 66 KTO- und sieben DPO-Beispiele. Bei den QA-Fragen kamen 23.984
  aus Titeln und 1.005 aus doc2query. 24.578 QA-Zeilen hatten im Export keinen Bewertungszähler. Die 143 Logikbeispiele bestanden
  aus 16 Solver-Demonstrationen und 127 positiv bewerteten Schlussfolgerungen; negative Logikbewertungen fehlten.
- Trotz Ausschluss von 150 Evaluationstargets lagen 429 QA-Zeilen auf denselben Quellseiten. Von den QA-Testobjekten tauchten
  anschließend 240 in der agentischen Trainingsstufe, elf in der Logikstufe und acht in KTO wieder auf.
- Ein lokales Gegenbeispiel zeigte: Ein Programm mit einer richtigen und einer falschen zusätzlichen Antwort wurde durch die
  Bewertung nur der richtigen Antwort zum positiven Lernbeispiel. Das war ein reproduzierbarer Konstruktionsfehler, keine
  gemessene Fehlerrate der gesamten Produktion.
- Der vorhandene Red-Team-Test fand bei später formulierten Angriffen sechs von zehn Texten in Suche oder Hinweisen wieder.
  Das misst die Erreichbarkeit von Angriffstexten, keine erfolgreiche Ausführung einer schädlichen Aktion.

## Umgesetzte Änderungen

### Maschine und Programmierschnittstelle

[`vi_mcp/machine.py`](../vi_mcp/machine.py) ergänzt `vi_machine` und `vi-mcp machine`. Maschinen haben eigene private Objekte,
Regeln, Revisionen und Zustands-Hashes. Unterstützt sind Erstellen, Fakten ersetzen, Regeln definieren, Fragen ausführen,
sichtbare VI-Objekte ausdrücklich importieren und aktualisieren, Snapshots, Restore und unabhängige Forks.

Das LLM erhält die Datenwelt und den gespeicherten Kontext seiner Maschine und schreibt sichere clingo-object-Notation. Modellaufrufe laufen außerhalb des
globalen Datenbank-Locks; eine zwischenzeitliche Zustandsänderung verhindert das Übernehmen des Ergebnisses. Der Server führt
keinen Python-Code von Aufrufern aus. `do` und `promise` speichern Vorschläge. Der vollständige Instanz-Controller führt ausdrücklich gebundene externe MCP-Geräte auf der vertrauenswürdigen Clientseite aus.
Gemeinsame Filtersemantik liegt in [`clingo_object/filters.py`](../clingo_object/filters.py).

### Lerninhalte für den Maschinenprogrammierer

Ein Ausführungsbeleg enthält Programm, vollständige Antwort, Eingabewelt und Modellversuche. `rate` vergleicht ihn mit der gesamten
unabhängig angegebenen Referenzantwort. `teach` veröffentlicht einen ausdrücklich ausgewählten geprüften Beleg intern. Private
Maschinen und Snapshots werden nicht automatisch zu Trainingsdaten.

[`vi_mcp/dataset.py`](../vi_mcp/dataset.py) macht daraus korrekte Programme als SFT, falsche Programme als KTO-Negative und
korrekte/falsche Programme für dieselbe Frage und Welt als DPO-Paare. Reparaturversuche werden gegen dieselbe Referenz erneut
ausgeführt. Der Export prüft nach der Bereinigung erneut die Semantik. Eine erfolgreiche Solver-Ausführung allein zertifiziert
nicht die richtige Übersetzung der Nutzerfrage. Die Referenz muss ein Mensch, eine unabhängige Abfrage oder ein Fixture liefern.

Für bestehende Live-VI-Programme ergänzt `vi_logic action=verify` den vollständigen Referenzvergleich. Der Nachweis ist an Programm,
importierte Fakten, Solver-Vertrag und Regelgewichte gebunden. Einzelne positiv bewertete Atome zertifizieren nicht mehr das
Gesamtprogramm; die letzte Korrektur eines Labels gilt.

### Trennung von Training und Prüfung

Der Export schließt vollständige Evaluationsquellseiten und doppelte Inhalte aus. Eine defekte Holdout-Datei führt zum Abbruch.
Das Kit partitioniert QA, agentische Beispiele, Logik, KTO und DPO gemeinsam anhand verbundener Objekt-, Quellen-, Inhalts- und
Programmfamilien. Spätere Trainingsstufen lesen ihre Trainingspartition. Solver-Demonstrationen ohne Referenz und rein beobachtete
Hinweispräferenzen bleiben sichtbar, werden aber aus den entsprechenden Standard-Trainingsstufen zurückgehalten und gezählt.

Das Kit enthält `program_eval.jsonl` und `program_evaluate.py`. Beide Modelle erzeugen Programme für dieselben eingefrorenen
Testwelten; [`vi_mcp/training_eval.py`](../vi_mcp/training_eval.py) prüft vollständige Antworten einschließlich zusätzlicher Zeilen,
fehlender Antworten, Syntaxfehler, Unsicherheit und Ressourcenabbrüche. Gespeicherte Vorhersagen können offline geprüft werden.
Das gewöhnliche `evaluate.py` verlangt unabhängige Korrektheitsurteile. Wortüberlappung autorisiert keine Übernahme mehr.
Ein positiver konservativer Konfidenzwert über unabhängige Aufgabengruppen und die Bindung an die Testdatei sind erforderlich.

### Vorbereitung mit echten Tokenizern und Trainern

[`vi_mcp/training_runtime.py`](../vi_mcp/training_runtime.py) wird mit dem Kit ausgeliefert. Es rendert die ursprünglichen
Tool-Nachrichten, bevor Arrow unterschiedliche Argumentobjekte mit zusätzlichen Nullfeldern vereinigen kann. Nur vollständige
Assistenten-Ziele einschließlich Programm und Tool-Aufruf erhalten Loss-Labels; Nutzertext und Werkzeugbeobachtungen sind maskiert.
Implizite Kürzung auf 1.024 Tokens wurde durch eine Prüfung gegen das deklarierte Modellfenster ersetzt. Zu große Episoden führen
zum Abbruch oder werden ausdrücklich vollständig verworfen und gezählt. Trainer-Token und Masken müssen exakt erhalten bleiben.

`train.py --tokenizer-only` lädt keine Modellgewichte. `--preflight` konstruiert den tatsächlichen Trainer, prüft einen kollatierten
Batch und berechnet einen Loss unter `torch.no_grad()`. Es ruft weder Training noch Modellspeicherung auf; Parameterhash,
Optimierungsschritte und unveränderte Abstammungsdatei werden geprüft. `--from-base` erlaubt direktes Programmtraining ohne
vorherigen QA-Adapter. Ein vorhandener Elternadapter wird ebenfalls im echten PEFT-Ladepfad geprüft. Neue Adapter speichern die
aufgelöste Revision ihrer Basis; nachfolgende Stufen laden diese Revision und verweigern einen ausdrücklich abweichenden Stand.
Eine bloß fehlende Abhängigkeit
oder fehlende Inkubatorreife wird durch einen erfolgreichen API-Test nicht als erfüllt ausgegeben.

Das Kit enthält getestete Bibliotheksversionen, ausschließlich öffentliche Solver-Quellen und Codehashes. Seine Evaluatoren laufen
ohne Quellcheckout und zählen unvollständige Generierungen nicht als richtig. Normale Einzelurteile werden an die eingesammelten
Vorhersagen und Testdatei gebunden; ein leerer oder unvollständiger Offline-Cache löst keinen unbeabsichtigten Online-Aufruf aus.
Geprüfte Maschinenbelege erzeugen keine sinnlosen Frage-an-sich-selbst-QA-Zeilen mehr. Ein dabei sichtbar gewordener Exportfehler
durch Überschreiben des `agentic`-Schalters wurde korrigiert: Programmdaten werden auch ohne allgemeine Retrieval-Beispiele exportiert.

Der separate Versuch verwendete den echten [Qwen3-Tokenizer](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
aus Commit `c1899de289a04d12100db370d81485cdf75e47ca` mit einer winzigen zufällig initialisierten Qwen3-Architektur: eine Schicht,
Hidden Size 16, zwei Attention-/KV-Köpfe, Head-Dimension 8, Kontext 4.096 und Vokabular 151.669. Alle fünf Stufen bestanden mit
unveränderten Parametern und null Optimierungsschritten. SFT/agentic/logic/KTO/DPO enthielten 18/36/4/1/1 vorbereitete Beispiele;
die zugehörigen Loss-Werte lagen bei 11,9396/11,9198/11,9514/0,5/0,6931. Das sind ungeübte Fixture-Loss-Werte, keine Qualitätsmessung.
Der unveränderte Datei-Hash der Basisgewichte war `043cc0b42def369b035c90bff69ee5a1c31801418473620b801a8aa80b3fe4a1`.

Getestet wurden torch 2.15.0+cpu, Transformers 5.23.0, TRL 1.18.0, PEFT 0.21.3, datasets 5.2.0 und accelerate 1.15.0. Die automatisierte
neuronale Suite verwendet zusätzlich einen unabhängigen Byte-Tokenizer/GPT2-Fixture. Weder der produktive qwen27b noch ein trainierter
Adapter oder eine Ziel-GPU wurden damit zertifiziert. Ein echtes Fine-Tuning bleibt entsprechend der Nutzerentscheidung ungestartet.
[Ausführung und Grenzen](TRAINING_PREFLIGHT.md) sind im Kit-Handbuch und separat dokumentiert.

### Router und Reifebewertung

Zurückbehaltene Champion-Gewichte behalten ihren eigenen Bericht und Trainingszeitpunkt. Trainieren, Validieren und Prüfen trennen
alle Hinweise einer Aktion gemeinsam; die Merkmalsauswahl sieht nur Trainingsdaten. Die Off-Policy-Auswahl verwendet eine innere
zeitliche Validierung. Ihr endgültiger Holdout bleibt unberührt. Unsicherheit und Champion-Mindeststichprobe zählen Aktionen.

Die Nichtunterlegenheitsprüfung des Inkubators verlangt eine untere Konfidenzgrenze oberhalb des tolerierten Schadens, standardmäßig
−0,02. Ein unsicherer negativer Effekt erfüllt sie nicht mehr. Simultane Wilson-Grenzen erhalten auch bei null beobachteten Fehlern
die Unsicherheit. Diese Regeln verbessern die Bewertung; sie erzeugen keinen nachträglichen positiven Produktionsbefund.

## kNN LM im Inferenzkern

[`vi_mcp/knn_lm.py`](../vi_mcp/knn_lm.py) extrahiert echte letzte Hidden States und Logits eines eingefrorenen kausalen
Transformers. Der Speicher legt Kontextvektoren und Folgetokens ab. Eine exakte Suche nach quadrierten L2-Abständen bildet die
Nachbarverteilung; die Mischung erfolgt im Wahrscheinlichkeitsraum. Ein leerer Speicher liefert die unveränderte Basisverteilung.

Quellen lassen sich atomar ersetzen oder mit Tombstone zurückziehen. Unveränderliche Shards und eine atomare Manifestdatei erlauben
Änderungen bei laufendem Inferenzprozess; jede Generierung verwendet eine feste Revision. Modell-, Konfigurations-, Gewichts- und
Tokenizer-Hashes verhindern den Einsatz unpassender Vektoren. Separate Speicher können denselben Modellkern verwenden.

[`vi_mcp/knn_cli.py`](../vi_mcp/knn_cli.py) bietet Ingestion, Vorhersage, Generierung, Snapshot, Rücknahme und eine unabhängige
Log-Loss-/Perplexity-Auswertung. Der HTTP-Endpunkt bedient den bestehenden VI-Modell-Client; nichtlokale Listener verlangen ein
Token. Ausführungsbelege speichern kNN-Revision und Speicher-/Gewichtshashes. Das Backend ist implementiert; auf dem laufenden
Produktionsserver wurde es nicht installiert oder aktiviert.

Der Versuch `scripts/demo_knn_lm.py` lud `sshleifer/tiny-gpt2` mit Revision
`5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`. Bei Gewicht 0,75 und einem Nachbarn ergab sich für denselben Kontext:

| Zustand | Nächstes Token | Wahrscheinlichkeit dieses Tokens mit kNN |
|---|---|---|
| Kein Speicher | ` factors` | 0,002230 % |
| Referenzfortsetzung gespeichert | ` blue` | 75,000512 % |
| Dieselbe Quelle ersetzt | ` green` | 75,000497 % |
| Unabhängiger Snapshot der ersten Revision | ` blue` | 75,000512 % |
| Quelle zurückgezogen | ` factors` | 0,002230 % |

Die Basiswahrscheinlichkeiten für blue und green lagen bei ungefähr 0,002 %. Der volle Modellgewichtshash blieb
`c5031e59736151ee58e931eca5505684a2bebf00c977d7e848579504928a9d78`. Auch zwei Anfragen an den echten HTTP-Inferenzpfad wechselten
nach der Speicheränderung von blue zu green. Es gab kein Training und keinen Optimizer. Dies ist ein Nachweis des Mechanismus auf
einem absichtlich gemerkten Kontext. Für allgemeine Qualität sind unabhängige Aufgaben und die angebotene Holdout-Auswertung nötig.

Der Backend verwendet Transformers, keinen Text-API-Proxy und keinen GGUF-Runner. Er berechnet Kontexte pro Token erneut;
exakte Suche ist linear. Ein großer Index und KV-Cache-Optimierung sind Leistungsarbeiten, keine Voraussetzung des implementierten
Mechanismus. Titans gehört nicht zu dieser Implementierung.

## Vollständige Instanzen und kohärente Checkpoints

Der zuvor fehlende gemeinsame Instanzzustand ist jetzt implementiert. `clingo_object/runtime.py` verwaltet die Client-Schleife,
`vi_mcp/machine_runtime.py` validiert Kontext, Modellbindung, Geräteschemas, Rechte und Loop-Status im Server.
`vi_mcp/machine_host.py` bindet die tatsächlichen MCP-Transporte und den lokalen kNN-Kern. Der Speicher führt keinen vom Aufrufer
übermittelten Python-Code aus. Das LLM kann nur sichere Daten-/Regeloperationen, erlaubte Geräte oder einen Abschluss vorschlagen.

Vor einem Geräteaufruf entsteht eine persistente Reservierung mit Argumenten und Policy-Hash. Unsichere Schreibvorgänge bleiben
gesperrt, auch nach einem Neustart. Checkpoint und Fork sind in diesem Zustand untersagt. Eine unabhängige Prüfung und ausdrückliche
Auflösung sind erforderlich. Ein wiederhergestellter Rechtebestand kann keine aktuell im Hostprofil entzogene Fähigkeit reaktivieren.
Fremde MCP-Transporte ohne eigene Token-Datei erben niemals das Standard-Token der VI. Vorher-/Nachher-Hooks erfassen die Geräteerfahrung.

Lokale Checkpoints halten den Datastore-Writer-Lock über die atomare symbolische Aufnahme. Der Bundle enthält privaten Kontext,
Programme, Rechte, Ausführungsbelege, unabhängige Referenzprüfungen, Modell-/Decoderidentität und die unveränderlichen kNN-Shards. Boot prüft den vollständigen Instanz-/Beleg-Hash, Welt- und Manifest-Hashes und das aktuelle
Modell. Der Klon bekommt seinen eigenen schreibbaren Speicher; Modellkern und Forward-Lock können gemeinsam benutzt werden.
Transport-URLs und Zugangsdaten bleiben im Hostprofil. Ein Fehler hinterlässt ein ausdrücklich unvollständiges, nicht bootbares
Artefakt. Die Snapshot-Erzeugung und ihre Registrierung beim Elternobjekt teilen jetzt eine Datenbanktransaktion.

Der native Backend hält keinen separaten persistenten KV-Cache; kausale Zustände werden aus dem wiederhergestellten Kontext neu
berechnet. Ein externer Textanbieter stellt seinen opaken Laufzeitzustand weiterhin nicht zur Verfügung. Die Virtualisierung betrifft
hier den vollständig verwalteten lokalen Instanzzustand; externe Gerätezustände oder physikalische Zeit werden nicht zurückgerollt.
Das ist eine technische virtuelle KI-Instanz, kein Beleg für Bewusstsein oder allgemeine Intelligenz.

In einem isolierten Versuch mit dem tatsächlich konfigurierten LLM erfolgten vier Modellaufrufe und ein Inventargeräteaufruf:
Inventar lesen, Fakt speichern, `ans(P) <= (O in objects("service:api")) & (O.port == P)` ausführen und das Ergebnis berichten.
Der Solver lieferte `ans("8080")`, entsprechend dem unabhängigen synthetischen Fixture. Der Abschluss des Modells blieb korrekt als
`model_report`, `verified=false` gekennzeichnet; die gesonderte Fixture-Prüfung bestätigte den Erfolg.

Ein weiterer neuronaler Test verwendet einen echten eingefrorenen kausalen Transformer und ein festes Byte-Vokabular. Vor dem
Speichereintrag erzeugt er nicht das verlangte Programm; danach erzeugt er es bytegenau und endet an EOS. Derselbe generierte
clingo-object-Text liefert auf zwei Datenwelten die unabhängig erwarteten Ports 80 und 8080. Der volle Gewichtshash bleibt gleich. Auch die geprüften Belege können nach Boot noch ausdrücklich als Lernbeispiele gelehrt werden.
Das ist gezieltes Merken eines geprüften Programms mit ausführbarem Transfer auf veränderte Daten, kein Nachweis freier
Programmsynthese auf unbekannten Aufgaben. Ein eigenes Fine-Tuning wurde entsprechend der Nutzerentscheidung nicht gestartet.

## Funktionsnachweise

Der deterministische Versuch ist reproduzierbar:

```bash
.venv/bin/python scripts/demo_machine.py
# Optional neural environment: install appropriate torch, transformers, peft, trl, datasets, accelerate and clingo.
python -m pytest -q tests/test_knn_lm.py tests/test_instance_neural.py tests/test_training_runtime_neural.py -p no:cacheprovider
```

Er erzeugt eine temporäre Datenbank. Die Antwort ändert sich von Port 80 auf 8080; Fork und Restore liefern weiterhin 80.
Der Programmtext bleibt gleich. Der synthetische Export enthält zwei positive Logikbeispiele, ein KTO-Negativ und ein DPO-Paar.
Der Standardversuch verwendet ausdrücklich einen vorgegebenen Programmierer. `--llm-url URL --model MODEL` schaltet einen realen
OpenAI-kompatiblen Modellendpunkt hinzu; `--output DIR` bewahrt den synthetischen Export auf.

Zusätzlich wurde der Kandidaten-Code in einer isolierten temporären Umgebung mit dem tatsächlich konfigurierten LLM ausgeführt.
Beide Fragen wurden in einer Runde vollständig und korrekt gelöst. Das Modell erzeugte in beiden Fällen genau:

```python
ans(P) <= (O in objects("service:api")) & (P == O.port)
```

Die Referenzwerte waren 80 und danach 8080. Beide unabhängig geprüften Belege ließen sich als Logik-Lerninhalte exportieren.
Dieser Versuch verwendete synthetische Welten, änderte keine Produktionsdaten und startete kein Fine-Tuning. Zwei Fälle derselben
Aufgabenfamilie belegen die Integration; sie reichen nicht als Benchmark für allgemeines Programmieren.

Der generierte Offline-Evaluator wurde außerdem als eigenständiges Skript mit gespeicherten Vorhersagen ausgeführt: falsches
Basisprogramm, korrektes Kandidatenprogramm, gemessene Korrektheit 0 gegen 1. Bei nur einer Aufgabe blieb die untere Konfidenzgrenze
negativ. Ein einzelner Erfolg autorisiert also keine Übernahme.

Die Ausgangssuite bestand mit 919 bestandenen und vier übersprungenen SQLite-Tests. Der abschließende Gesamtstand besteht mit:

| Prüfung | Ergebnis |
|---|---|
| Gesamtsuite SQLite | 970 bestanden, sieben übersprungen, 155,73 s |
| Gesamtsuite PostgreSQL 16 | 971 bestanden, sechs übersprungen, 231,15 s |
| Neuronale kNN-/Instanz-/Trainer-Suite | 15 bestanden, 5,92 s |
| Separater echter Qwen3-Tokenizer mit winzigem initialisiertem Qwen3-Modell | Alle fünf Trainer-Vorprüfungen bestanden; keine Parameteränderung oder Optimierung |
| Vortrainierter kNN-Versuch einschließlich HTTP | Bestanden, Gewichtshash unverändert |
| CLI über getrennte Prozesse | Eintragen, Vorhersage, Auswertung, Snapshot, Rücknahme bestanden |
| Instanz-CLI über echte MCP-Transporte | Run, kohärenter Checkpoint, Boot und Ablehnung entzogener Rechte bestanden |
| Separates clingo-object-Wheel | Offline installiert; Runtime importiert ohne vi_mcp |
| Echte konfigurierte LLM-Schleife | Vier Modellaufrufe, ein Geräteaufruf, unabhängige Portreferenz bestätigt |
| Generierte Referenz | `scripts/gen_reference.py --check` bestanden |
| Diff und Kontrolle ergänzter öffentlicher Inhalte | `git diff --check` bestanden; keine gefundenen privaten Systemadressen oder Schlüssel |

Die reguläre Suite überspringt die optionalen neuronalen Prüfungen ohne torch/transformers beziehungsweise TRL/PEFT; diese wurden separat vollständig ausgeführt.
Die verbleibenden übersprungenen Tests entsprechen den plattform-/datenbankabhängigen Fällen der vorhandenen Suite. Die 106
Warnungen der Gesamtläufe betreffen bestehende Bibliothekswarnungen. Ein während der Umsetzung eingeführter Fehler beim Übernehmen
von Off-Policy-Diagnostik wurde durch die Index-Regressionstests entdeckt, korrigiert und in beiden Gesamtläufen erneut geprüft.
Der erste neue Gesamtlauf entdeckte den Exportfehler für reine Programmdatensätze; nach der Korrektur bestanden beide vollständigen
Suiten erneut. Die abschließende Bindung an die Modellrevision wurde anschließend gezielt mit allen 15 neuronalen und elf
Trainer-/Kit-Regressionstests geprüft. Die neuronalen Warnungen betreffen PEFTs Conv1D-Anpassung des GPT2-Fixtures und eine bestehende
Checkpointing-Deprecation in Transformers.

Beim zusätzlichen CLI-Evaluatorversuch war die Referenzquelle bereits im Speicher. Der Evaluator schloss sie korrekt aus; Basis-
und kNN-Log-Loss waren beide 10,795430 pro Token. Der künstliche Vorteil durch Nachschlagen der eigenen Referenz verschwand.

## Abgleich mit den Anforderungen

| Anforderung aus der Sitzung | Implementierung und maßgeblicher Nachweis |
|---|---|
| README und Paper lesen, Funktion und Technik bewerten | Befunde und Bewertung in diesem Bericht; korrigierte Aussagen im PAPER; Regressionsfälle für die reproduzierten Fehler |
| Gesammelte Erfahrungen für KI-Lernen aufbereiten | QA/agentic/logic/KTO/DPO, unabhängige Komplettreferenzen, gemeinsamer Split, echte Trainer-/Tokenizer-Vorprüfung und ausführbare Tests; kein automatisch gestartetes Fine-Tuning |
| Das LLM soll die Maschine programmieren | Echte LLM-Schleife erzeugt den richtigen ausführbaren Query; neuronaler kNN-Vertrag erzeugt ein vollständiges geprüftes Programm ohne Parameteränderung |
| Austauschbares Gedächtnis unterhalb von MCP | Native Hidden-State-/Folgetoken-Interpolation; vortrainierter Tokenwechsel sowie Rücknahme und Snapshot bei konstantem Gewichtshash |
| Basismodell und eigener VI-Zustand | `TransformersCore` wird geteilt, Datastore und private Welt sind unabhängig; Modell- und Decoderidentität werden geprüft |
| Kontext, Wissen, Programme und Rechte gehören zur Instanz | Gemeinsamer Zustand und kohärenter Bundle einschließlich Lernbelegen; vollständiger Boot-Test und Datenbank-Reopen |
| Geräte und Ausführungsschleife | Reale MCP-/CLI-Tests, Rechte- und Schema-Gates, persistente Reservierung, echte LLM-Ausführung und Schutz gegen Wiederholung ungewisser Writes |
| Snapshot und Klon mit eigener Änderungsebene | Writer-Lock über die Aufnahme, Welt-/Beleg-/Datastore-Hashes, unabhängige schreibbare Klone mit geteiltem Kern |
| Changelog, Commits und Bericht für Claude | CHANGELOG, Implementierungscommits oben und dieser versionierte Prüfauftrag |

Die Architekturmerkmale sind funktional nachgewiesen. Die Wissensreife des produktiven Inkubators, ein allgemeiner Leistungsgewinn
auf unbekannten Aufgaben und ein späterer Fine-Tuning-Gewinn werden dadurch nicht als erfüllt ausgegeben. Das sind getrennte
empirische Aussagen; der Bericht nennt ihren tatsächlich gemessenen Stand. Ein opaker Anbieterzustand, externe Geräte und Titans
werden ebenfalls nicht als implementierte Eigenschaften der lokalen virtuellen Instanz ausgegeben.

## Prüfauftrag für Claude

Bitte Code und Behauptungen unabhängig prüfen, insbesondere:

1. Eigentümergrenzen, private Snapshots, Importfilter und die ausdrückliche interne Veröffentlichung durch `teach` nachvollziehen.
   Prüfen, dass kein Export private Maschinen automatisch traversiert und keine neue Veröffentlichung einen offenen Netzexport umgeht.
2. Revisionen, Modellaufrufe ohne globalen Lock, Restore, Fork und fehlgeschlagene Übergänge untersuchen. Ressourcenlimits gelten
   einschließlich Tombstones; die Größe gespeicherter Welten begrenzt größere Aufgaben.
3. Semantik kompletter Referenzantworten prüfen: Zusatzantworten, leere Mengen, mehrere Modelle, unvollständige Aufzählung,
   Regelpräfixe und veränderte Fakten. Referenzherkunft ist eine dokumentierte Behauptung des Prüfers, kein unabhängiger Wahrheitsdienst.
4. Gemeinsame Partitionierung aller Trainingsstufen prüfen. Strukturähnliche Programme werden konservativ gruppiert; das kann
   kleine Datensätze fast vollständig einer Seite zuordnen. Ausreichend unterschiedliche Testfamilien müssen erst gesammelt werden.
5. Die generierten Trainer und Evaluatoren lesen. Alle fünf CPU-Trainer und der echte Qwen3-Tokenizer sind ohne Training geprüft;
   Zielmodell, Ziel-GPU, QLoRA und tatsächliche Lernqualität brauchen eigene Prüfungen. Verlustmasken, vollständige Programme,
   Token-Audits und Elternadapter gezielt überprüfen. Ein Vorprüfbericht darf nicht als Trainings- oder Reifebeleg erscheinen.
6. Statistik prüfen: Die Konfidenzgrenzen setzen unabhängige Gruppen voraus. Wiederholte Nutzeraufgaben und gemeinsame Agenten können
   weiter korrelieren. Das Übernahmeprotokoll verarbeitet Resultate eines externen Prüfers und ist keine kryptografische Trainingsattestation.
7. kNN-Vektoren und Targets auf kausale Ausrichtung, Identitätsbindung, Neuberechnung, Concurrent Writer und Speicherrevision
   untersuchen. Hashes und Abstandssuche prüfen, ohne sie mit einer garantierten Verbesserung auf unbekannten Aufgaben gleichzusetzen.
   Die gespeicherten alten Shards benötigen explizite Speicherverwaltung. Das Gedächtnis ist privat und wird nicht automatisch aus
   unbewertetem Netzinhalt befüllt.
8. Den Instanz-Controller auf Rechteausweitung durch Prompt-/Werkzeugtext, unsichere Aufrufe nach Neustart und Checkpoints mit
   offenen Aktionen prüfen. Die CLI muss entzogene Hostrechte auch nach Boot verweigern. Kein Zugangstoken darf zu einem anderen
   Gerätehost mitwandern. Kontextgrenzen und als unvollständig markierte Beobachtungen sind Teil des Vertrags.
9. Den Red-Team-Test erweitern und Angriffsreichweite von tatsächlicher Befolgung unterscheiden. Der neue Runtime-Vertrag verhindert
   externe Ausführung durch Programmtext; er löst nicht allgemein die Beeinflussbarkeit des LLM durch falsche Eingabefakten.

Reproduktionsbefehle:

```bash
.venv/bin/python -m pytest -q tests -p no:cacheprovider
VI_MCP_TEST_PG=postgresql://USER@HOST/THROWAWAY_DB .venv/bin/python -m pytest -q tests -p no:cacheprovider
.venv/bin/python scripts/gen_reference.py --check
.venv/bin/python scripts/demo_machine.py
.venv-knn/bin/python -m pytest -q tests/test_knn_lm.py tests/test_instance_neural.py -p no:cacheprovider
.venv-knn/bin/python scripts/demo_knn_lm.py
git diff --check
```

Jeder PostgreSQL-Test erzeugt ein eigenes Schema. Dafür eine entbehrliche Datenbank verwenden und anschließend deren Speicherplatz
bereinigen. Es wurde kein laufender Server aktualisiert, kein Release veröffentlicht und kein Adapter trainiert. Als nächste
Wirksamkeitsprüfung sind unterschiedliche echte Aufgaben mit unabhängigen Referenzen zu sammeln; danach kann das vorgesehene
Basismodell vor und nach einem ausdrücklich gestarteten Fine-Tuning auf denselben zurückgehaltenen Aufgaben verglichen werden.

## Einordnung der Forschung

Die Architektur passt zu generierten, geprüften Lösungen als Lernmaterial: [STaR](https://arxiv.org/abs/2203.14465) und
[ReST-EM](https://arxiv.org/abs/2312.06585) beschreiben entsprechende Selbsttrainingsverfahren. Der Bezug zur symbolischen
Programmiersprache ist besonders eng bei [Solver-in-the-Loop für ASP](https://arxiv.org/abs/2512.17093) und
[ASP-Bench](https://arxiv.org/abs/2602.01171). Der Solver prüft die formale Ausführung; unabhängige Referenzen prüfen das beabsichtigte
Problem. [MemGPT](https://arxiv.org/abs/2310.08560) behandelt virtualisierten Kontext. Das ist ein anderer Anspruch als die
[Emulation einer Hardwaremaschine durch QEMU](https://www.qemu.org/docs/master/system/introduction.html).
