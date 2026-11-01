# Agritech Answers — introduction au dossier technique

*Ce document ouvre le dossier. Les quatre parties qui suivent sont les rapports rédigés à chaque étape du projet, dans l'ordre où ils ont été produits.*

---

## Résumé

Agritech Answers estime le rendement d'une parcelle agricole et compare six cultures sur leur marge brute. Huit informations suffisent à l'interroger ; le modèle en attend treize, les cinq autres étant calculées.

Après préparation, le jeu consolidé compte **999 769 parcelles** et 13 variables explicatives, découpées en 70 / 15 / 15. Quatre modèles ont été comparés sur 100 essais d'optimisation chacun. C'est le plus simple qui l'emporte : une **régression régularisée**, retenue pour son erreur, sa rapidité, sa légèreté et la lisibilité de ses coefficients. Sur le jeu de test, ouvert une seule fois après le choix : **0,40 t/ha** d'erreur absolue moyenne, soit 80 €/ha à 200 €/t, et 91 % de la variation du rendement expliquée.

Le résultat central va à l'encontre de la demande. **Les six cultures ne se départagent pas sur le rendement** : à conditions de parcelle identiques, l'écart entre la meilleure et la pire vaut 6,5 kg/ha, quinze fois moins que le seuil de pertinence pratique retenu. Quatre variables portent la totalité du signal — la pluviométrie, puis la fertilisation, l'irrigation et la température ; les neuf autres pèsent quelques kilogrammes par hectare.

Le produit en tire sa forme : une **API de six routes** servant un artefact scellé, et une **interface** qui compare les cultures non sur leur rendement mais sur leur **marge brute**, calculée à partir de prix de référence sourcés et de charges saisies par l'agriculteur.

Une réserve traverse tout le dossier : les données de parcelles sont **simulées**. Le prototype démontre la chaîne complète ; ses chiffres ne sont pas des résultats agronomiques transposables.

---

## La demande

Agritech Answers accompagne des exploitations agricoles dans leurs décisions de culture. La demande portait sur deux questions qu'un agriculteur se pose avant de semer :

- **quel rendement attendre** d'une parcelle, connaissant ses conditions et ses pratiques ;
- **quelle culture choisir** parmi celles qu'il peut implanter.

Le produit attendu est une application utilisable par un agriculteur — pas un modèle dans un notebook. La chaîne complète devait donc être construite : préparation des données, entraînement, mise en service, interface, et l'outillage qui permet de tout rejouer.

---

## Les données fournies

Deux jeux, et une difficulté qui a orienté tout le projet.

| Jeu | Volume | Ce qu'une ligne décrit |
|---|---|---|
| Parcelles | 1 000 000 | une parcelle : culture, climat, pratiques, rendement |
| Référentiel FAO | 56 717 | un pays, une culture, une année |

**Les deux ne se joignent pas.** Le fichier des parcelles ne contient ni pays ni année ; ses « régions » sont quatre zones génériques sans correspondance géographique. La seule information commune aux deux jeux est le type de culture.

Cette contrainte n'est pas un détail d'implémentation : elle détermine ce que le produit peut affirmer, et ce qu'il doit se refuser à affirmer. La première partie de ce dossier lui est consacrée.

---

## Les quatre parties

| Partie | Question traitée | Ce qu'elle établit |
|---|---|---|
| **1. Préparation des données** | que contiennent réellement ces fichiers, et que peut-on en construire ? | un jeu consolidé de 999 769 parcelles et 14 variables, et le constat que les six cultures ne se distinguent pas |
| **2. Modélisation** | quel modèle, quelle performance, quels leviers ? | une régression régularisée, 0,40 t/ha d'erreur moyenne, et quatre leviers chiffrés |
| **3. L'API de prédiction** | comment servir ce modèle, et avec quelles garanties ? | un service de six routes, son contrat, et ce qu'il refuse d'affirmer |
| **4. L'interface** | que voit l'agriculteur, et que l'empêche-t-on de croire ? | deux parcours, et la marge brute qui départage là où le rendement ne départage pas |

Chaque partie reprend les limites annoncées par la précédente et dit ce qu'elle en a fait. C'est le fil le plus utile pour suivre le dossier : une limite énoncée à l'étape 1 se retrouve traitée, contournée ou assumée à l'étape 3.

---

## Comment lire ce dossier

**Ces documents sont datés.** Chacun a été rédigé à la fin de son étape et décrit l'état du projet à ce moment-là. Leurs chiffres — nombre de tests, taille de l'image, périmètre couvert — sont des instantanés justes à leur date, pas l'état actuel. Ils n'ont pas été réécrits après coup : un dossier dont les conclusions bougeraient rétrospectivement ne vaudrait rien.

**Les encadrés « Suite donnée » font le lien.** Là où une étape ultérieure a levé une limite, un encadré le signale à l'endroit exact où elle était énoncée. Le lecteur n'est donc jamais laissé sur une réserve périmée, sans que le texte d'origine ait été altéré.

**Les parties 3 et 4 se lisent à deux niveaux.** Leurs premières sections décrivent ce que l'agriculteur voit et ce que l'application affirme — elles se lisent sans connaissance technique. Les dernières portent sur le déploiement, la traçabilité et les tests, et s'adressent à qui reprendra le service.

---

## Ce qui ne figure pas ici

**Le rapport métier.** Un document distinct, plus court, s'adresse aux agriculteurs : résultats principaux, recommandations agronomiques et explication des variables clés, sans appareil technique. Le présent dossier en est le support de preuve.

**L'étape 5 — industrialisation.** La chaîne MLOps, l'intégration continue et la livraison de l'image n'ont pas donné lieu à un rapport : elles relèvent du développement, et sont documentées dans le `README.md` du dépôt. On y trouve l'orchestration des quatre étapes de la chaîne, les empreintes qui garantissent la cohérence entre données, modèle et artefact servi, les contrôles exécutés à chaque poussée, et la publication de l'image.

**Une réserve qui traverse tout le dossier.** Les données de parcelles présentent toutes les caractéristiques d'un jeu **simulé** : variables parfaitement uniformes, modalités rigoureusement équilibrées, relations d'une régularité qu'on n'observe jamais en agriculture réelle. Le prototype démontre la chaîne complète, de la donnée à l'application. Mais les effets chiffrés qu'il contient décrivent le comportement de ce jeu de données, et **ne sont pas des résultats agronomiques transposables** à des parcelles réelles.
