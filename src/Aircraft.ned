package aerocomsystemproject;

module Aircraft
{
    parameters:
        int nDL = default(2);//(Cost)
        double k @unit(s); // Ogni quanto genera i pacchetti (VA)
        double m @unit(s); //Tempo di monitoraggio (Cost)
        string distribution;
       

    gates:
        output out[nDL];


    submodules:
        packetGenerator: PacketGenerator {
            // @display("p=64,160");
            // @display("p=201,172");
        }
        linkSelector: LinkSelector {
            @display("p=57,37");
        }
    connections:
        for j=0..nDL-1 {
            linkSelector.out[j] --> out[j];
        }
        
        packetGenerator.out --> linkSelector.in;
}
