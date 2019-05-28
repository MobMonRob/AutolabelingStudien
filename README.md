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
    + __data_generation &rarr;__ *Enhält Klassen zum Generieren von eigenen Markerdaten. Bei der weiteren Verwendung des Repositories kann dieser Ordner ignoriert werden. (kurze Anleitung in der Testklasse)*
    
    + __datavec &rarr;__ *Enhält Implementierungen der Schnittstellen von Datavec*
        + __JsonTrialRecordReader &rarr;__ *RecordReader, der die Daten der eines Trials im JSON-Format einliest. Jeder Record repräsentiert einen Frame eines Trials*
        + __RandomizedRecordReader &rarr;__ *Erbt von JSONTrialRecordReader. Mischt die Reihenfolge der Frames vor der Ausgabe. Dadurch konnten bessere Ergebnisse erzielt werden.*
        + __SequentialMarkerwiseTrialRecordReader &rarr;__ *Ausgabe einer Sequenz von Markerpositionen zur Verwendung in einem rekurrenten neuronalen Netz.*
        + __TrialFileIterator &rarr;__ *Helferklasse, die in den Implementierungen der RecordReader zum Iterieren über die Dateien verwendet wird.*
        
    + __preprocess_data &rarr;__ *Dieser Ordner enthält alle Klassen, die zum Parsen, Weiterverarbeiten und Konvertieren der JSON-Daten in den RecordReadern verwendet werden.*

        *Der komplette Vorverarbeitungsprozess wird in der Klasse __TrialDataManager__ definiert. Durch die Verwendung des [Strategie-Patterns](https://de.wikipedia.org/wiki/Strategie_(Entwurfsmuster)) können alle Bestandteile des Vorverarbeitungsprozesses ausgetauscht werden. Durch die Schnittstellen können zudem relativ einfach neue Strategien entwickelt und eingebunden werden.*
        + __builders &rarr;__ *Enhält [Builder](https://de.wikipedia.org/wiki/Erbauer_(Entwurfsmuster))-Klassen, zum übersichtlicheren Initialisieren der zentralen Klassen der Vorverarbeitung.*
        + __data_manipulation &rarr;__ *Enhält alle Manipulatoren und die Schnittstelle zum Definieren neuer Manipulationsstrategien.*
        + __data_model &rarr;__ *Enthält das Datenmodell für die Vorverarbeitung. (Marker, Frame...)*
        + __data_normalization &rarr;__ *Enhält alle Strategien zur Normalisierung der Markerdaten und die Strategie-Schnittstelle*
        + __labeling &rarr;__ *Enhält alle Strategien zum Labeling der Markerdaten und die Schnittstelle zum Definieren neuer Labeling-Strategien*
        + __preprocessors &rarr;__ *Enhält Preprocessor-Klassen, die den kompletten Vorverarbeitungsprozess durchführen und die fertigen Daten in ein Verzeichnis ablegen. (Wichtig zur Performance-Optimierung, da die Vorverarbeitungsschritte nur einmal durchgeführt werden müssen)*
    + __test &rarr;__ *Dieser Ordner enthält lauffähige Klassen zur Durchführung des Trainings. Zudem werden verschiedene Netzwerk-Konfigurationen definiert*    
        + __execution &rarr;__ *Enhält Klassen zum automatischen Trainieren von vielen verschiedenen Netzwerk-Konfiguration. Bei der weiteren Verwendung dieses Repositories kann dieser Ordner ignoriert werden.*
        + __tests &rarr;__ *Testklassen der unterschiedlichen Lösungsansätze*
            + __lstm &rarr;__ *Alle Tests, die auf LSTMs basieren.*
            + __marker_distance_labeling &rarr;__ *Tests, die die Distanzen der Marker als Features verwenden.*
            + __one_marker_labeling &rarr;__ *Alle Tests, bei dem sich ein neuronales Netz auf einen Marker spezialisiert. (Multilayer Perceptron und CNN)*
            
####Hinweise zum Training:
1. Pfade in den Testdateien müssen entsprechend angepasst werden.
2. Nur JSON-Dateien eines bestimmten Formats können verwendet werden.
3. Achtung vor NULL-Markern in den JSON-Dateien. Mit diesen ist kein Training möglich!
4. In der pom.xml kann festgelegt werden, ob mit der CPU oder GPU trainiert wird. (siehe Kommentare)

##Ausblick
Durch die Studienarbeit wurde eine Basis zum Einlesen und zur Vorverarbeitung von Markerdaten aus dem Motion Capturing geschaffen. Die Schritte der Vorverarbeitung können durch die definierten Schnittstellen angepasst und erweitert werden.<br>
In der Studienarbeit wurden zudem verschiedene Architekturen von neuronalen Netzen und verschiedene Arten von Trainingsdatensätzen getestet. Auf Basis dieses Repositories und der darin bereitgestellten Architekturen können weitere Tests auf besseren Rechnern durchgeführt werden. Vor allem bei LSTMs und sequentiellen Datensätzen besteht Potenzial zur Verbesserung.

        