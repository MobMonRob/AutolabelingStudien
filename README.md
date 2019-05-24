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
        
    + __preprocess_data &rarr;__ *Dieser Ordner enthält alle Klassen, die zur Parsen, Weiterverarbeiten und Konvertieren der JSON-Daten in den RecordReadern verwendet werden.*

        *Der komplette Vorverarbeitungsprozess wird in der Klasse __TrialDataManager__ definiert. Durch die Verwendung des [Strategie-Patterns](https://de.wikipedia.org/wiki/Strategie_(Entwurfsmuster)) können alle Bestandteile des Vorberarbeitungsprozesses ausgetauscht werden. Durch die Schnittstellen können zudem relativ einfach neue Strategien entwickelt und eingebunden werden.*
        + __builders &rarr;__ *Enhält [Builder](https://de.wikipedia.org/wiki/Erbauer_(Entwurfsmuster))-Klassen, zum übersichtlicheren Initialisieren der zentralen Klassen der Vorverarbeitung*
        + __data_manipulation &rarr;__ *Enhält alle Manipulatoren und die Schnittstelle zum definieren neuer Manipulationsstrategien.*
        + __data_model &rarr;__ *Enthält das Datenmodell für die Vorverarbeitung. (Marker, Frame...)*
        + __data_normalization &rarr;__ *Enhält alle Strategien zur Normalisierung der Markerdaten und die Strategie-Schnittstelle*
        + __labeling &rarr;__ *Enhält alle Strategien zur Labeling der Markerdaten und die Schnittstelle zum definieren neuer Labeling-Strategien*
        + __preprocessors &rarr;__ *Enhält Preprocessor-Klassen, die den kompletten Vorverarbeitungsprozess durchführen und die fertigen Daten in ein Verzeichnis ablegen. (Wichtig zur Performance-Optimierung, da die Vorverarbeitungsschritte nur einmal durchgeführt werden müssen)*
    + __test &rarr;__ *Dieser Ordner enthält laufähige Klassen zur Durchführung des Trainings. Zudem werden verschiedene Netzwerk-Konfigurationen definiert*    
        + __execution &rarr;__ *Enhält Klassen zum automatischen Trainieren von vielen verschiedenen Netzwerk-Konfiguration. Diesen Ordner bitte komplett ignorieren.*