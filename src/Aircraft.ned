package aerocomsystemproject;

module Aircraft
{
    parameters:
        double t @unit(s); // Ogni quanto il DL seleziona la nuova capacit� target
        int dimPoolMax; //Estremo superiore capacit� target
        int dimPoolMin; //Estremo inferiore capacit� target
        int operationMode; //0 o 1
        double X @unit(s); //Malus dovuto al monitoraggio
        double s @unit(byte); //Dimensione pacchetto
        string tDistribution = default("exponential");
        int nDL = default(2); //Numero di datalink
        double k @unit(s); // Ogni quanto genera i pacchetti 
        double m @unit(s); //Tempo di monitoraggio della capacit� dei datalink
        int DLMaxCapacity; //Id del datalink con capacit� maggiore
        bool trasmitting; //Il datalink sta trasmettendo oppure no

        @display("bgb=461,189");
    gates:
        output out[nDL];

    submodules:
        packetGenerator: PacketGenerator {
            @display("p=390.825,96.5;");
        }
        linkSelector: LinkSelector {
            @display("p=260.55,79.13;");
        }
        dataLink[nDL]: DataLink { }
    connections:
        for j=0..nDL-1 {
            linkSelector.out[j] --> dataLink[j].in;
        }

        packetGenerator.out --> linkSelector.in;

        for j=0..nDL-1 {
            dataLink[j].out --> out[j];
        }
}
