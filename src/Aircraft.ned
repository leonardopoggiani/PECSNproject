package aerocomsystemproject;

module Aircraft
{
    parameters:
        double t @unit(s); // Ogni quanto il DL seleziona la nuova capacità target
        int dimPoolMax; //Estremo superiore capacità target
        int dimPoolMin; //Estremo inferiore capacità target
        int operationMode; //0 o 1
        double X @unit(s); //Malus dovuto al monitoraggio
        double s @unit(byte); //Dimensione pacchetto
        string tDistribution = default("exponential");
        int nDL = default(2); //Numero di datalink
        double k @unit(s); // Ogni quanto genera i pacchetti 
        double m @unit(s); //Tempo di monitoraggio della capacità dei datalink
        int DLMaxCapacity; //Id del datalink con capacità maggiore
        bool trasmitting; //Il datalink sta trasmettendo oppure no

        @display("bgb=461,189");
    gates:
        output out[nDL];


    submodules:
        packetGenerator: PacketGenerator {
            @display("p=390.825,96.5;i=device/pc4");
        }
        linkSelector: LinkSelector {
            @display("p=260.55,79.13;i=abstract/router");
        }
        dataLink[nDL]: DataLink {
            @display("i=block/mac;p=93.605,79.13");
        }
    connections:
        for j=0..nDL-1 {
            linkSelector.out[j] --> dataLink[j].in;
        }

        packetGenerator.out --> linkSelector.in;

        for j=0..nDL-1 {
            dataLink[j].out --> out[j];
        }
}
