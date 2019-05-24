# AutolabelingStudien
Dieses Repository enthält den Code zur Studienarbeit: "Autolabeling basierend auf Deep Learning".

##Getting Started
Zur Verwaltung der Dependencies des Projekts wird Maven verwendet. Zum Öffnen des Projekts in Netbeans kann folgender Guide verwendet werden: [Netbeans Wiki: Open existing Project](http://wiki.netbeans.org/MavenBestPractices#Open_existing_project).

__Hinweis:__ Es kann zu Problemen mit der Java-Version kommen. Daher sollte vor dem Kompilieren überprüft werden, ob die Target-JDK richtig gesetzt ist. ([Setting Target SDK](https://blogs.oracle.com/roumen/netbeans-quick-tip-1-setting-target-jdk)). Zudem sollten die Project-Settings überprüft werden.
Empfohlene Java-Version: 8+.

##Aufbau des Projekts
Das Projekt wird mit der Java-Bibliothek __DL4J__ (Deep Learning for Java) umgesetzt. Zum Einlesen der JSON-Dateien wird die Bibliothek __Datavec__ verwendet, die Bestandteil von DL4J ist. 

###Projektstruktur

+ __src/main/java__
    + __data_generation &rarr;__ *Enhält Klassen zum Generieren von eigenen Markerdaten. Bei der weiteren Verwendung des Repositories kann dieser Orner ignoriert werden. (kurze Anleitung in der Testklasse)*
    + __datavec &rarr;__ *Enhält Implementierungen der Schnittstellen von Datavec*
        + __JsonTrialRecordReader &rarr;__ *RecordReader, der die Daten der eines Trials im JSON-Format einliest. Jeder Record repräsentiert einen Frame eines Trials*
        + __RandomizedRecordReader &rarr;__ *Erbt von JSONTrialRecordReader. Mischt die Reihenfolge der Frames vor der Ausgabe. Dadurch konnten bessere Ergebnisse erzielt werden.*
        + __SequentialMarkerwiseTrialRecordReader &rarr;__ *Ausgabe einer Sequenz von Markerpositionen zur Verwendung in einem rekurrenten neuronalen Netz.*
        + __TrialFileIterator &rarr;__ *Helperklasse, die in den Implementierungen der RecordReader zum Iterieren über die Dateien mit Trainingsdaten verwendet wird.*
        
        