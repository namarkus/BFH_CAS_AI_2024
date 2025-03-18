# Project02 - Deep Reinforcement-Learning mit torchrl

Das Projekt BFH-Digital-Marketing ist im Rahmen des 2. Semesterprojekts
des CAS AI Herbst 2024 an der BFH entstanden.

Folgende Personen der _Gruppe 3_ haben zu diesem Projekt beigetragen:

- Hans Wermelinger
- Helmut Gehrer
- Markus Näpflin
- Nils Hryciuk
- Stefano Mavilio

##  Ausführung der Anwendung

### Vorbedingungen

Installation des kompletten Conda Environments:

```bash
conda install -f environment.xml
```

Wechsel in das Environment

```bash
conda activate bfh_ai_rl
````

### Starten des DRL

```bash
# Wechseln in das Projektverzeichnis cd Project02/Grp3
python digital_marketing.py
```

### Monitoring

```bash
# Wechseln in das Projektverzeichnis cd Project02/Grp3
tensorboard --logdir p02_metrics
```

Unter Umständen von einem anderen Conda Environment nötig, da mit aktuellem noch Kompatibilitäts-Issues bestehen.

## Felder und ihre Bedeutung

| Feldname | Typ (Range) | Berechnung | Beschreibung |
| -------- | ----------- | ---------- | ------------ |
| Keyword  | str         | aus adwords.csv geladen | Das Keyword selbst |
| Generation | int (0+)  | +1 bei jeder Epoche | Generation des Keywords im aktuellen Lauf |
| Competitiveness | float (0.00-1.00) | Initial aus CSV; wird pro Generation im Rahmen der volatility ramdom verändert |  Wettbewerbsfähigkeit (je höher, desto einfacher ) |
| DifficultyScore | int (0-100) | ???? | Competitiveness in Bezug auf das gesamte Suchvolumen (< 30 guter Organic rank; > 70 schlechter Organic Rank) |
| OrganicRank | int (0-100) | Competitiveness * 100 * (+/-  volatility) | Organisches Ranking in den Suchergebnissen |
| OrganicClicks | int (0+) | Inital aus CSV; danach durch Analytics-Simulation aktualisiert | Klicks als direktes Resultat der Suche |
| OrganicConversions | int (0+) | Inital aus CSV; danach durch Analytics-Simulation aktualisiert | Summe aller Abschlüsse aus Sucheresultaten |
| OrganicClickThroughRate | float (0.00 - 1.00) | 100 / OrganicClicks * OrganicConversions | Abschlüsse aus Suchresultaten; je höher je besser |
| AdContingent | int (0-10) | Initial aus CSV; danach durch Actions erhöht, bzw. durch Analytics wieder gesenkt | Kontingent der gekauften aber noch nicht geklickten Ads |
| AdImpressionShare | float (0.00 - 10.00) | Contingent * (1 - Competitiveness) * (+/-) volatility | Anzahl, wie oft Inserat angezeitgt worden ist. |
| CostPerClick | float (0.00+) | Inital aus CSV; Wenn 0.0, dann Random im Rahmen von 0.20 - 4.00; wird pro Generation im Rahmen der volatility ramdom verändert | Kosten pro AdClick |
| AdSpent | float (0.00+) | Initial aus CSV; danach durch Analytics-Simulation aktualisiert | Summe aller Ausgaben für dieses Keyword |
| AdClicks | int (0++) | Initial aus CSV; danach durch Analytics-Simulation aktualisiert | Summe aller bezahlten Clicks (paid_clicks) |
| AdConversions | int (0++) | Initial aus CSV; danach durch Analytics-Simulation aktualisiert | Summe aller Abschlüsse aus bezahlten Clicks |
| AdClickThroughRate | float (0.00 - 1.00) | 100 / AdClicks * AdConversions | Abschlüsse aus bezahlter Werbung; je höher je besser (paid_CTR) |
| ReturnOnAddSpent | float (0.00+) | 100 / AdConversions * Ertrag * AdSpent | Ertrag pro Investition in ein einzelnes Keyword |
| OtherClicks | int (0+) | Inital aus CSV; danach durch Analytics-Simulation aktualisiert | Klicks als anderen Quellen (Website, Bookmarks, andere Suchmaschinen |
| OtherConversions | int (0+) | Inital aus CSV; danach durch Analytics-Simulation aktualisiert | Summe aller Abschlüsse aus anderen Quellen |
| OtherClickThroughRate | float (0.00 - 1.00) | 100 / OtherClicks * OtherConversions | Abschlüsse aus anderen Quellen; je höher je besser |
| ConversionRate | float (0.00+) | (OrganicClicks + AdClicks + OtherClichs) / (OrganicConversation + AdConversions + OtherConversions) | ConversionRate über alles |
| CostPerAcquisition | float (0.00+) | AdSpent über alles / (OrganicConversation + AdConversions + OtherConversions)| Gesamtkosten der Aquisition |
| Monitor | boolean (0/1) | Fix aus csv | Steuert die Ausgabe in Log und Metrics pro Keyword |



