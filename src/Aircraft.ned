package aerocomsystemproject;

module Aircraft
{
    parameters:
        int nDL = default(2);//(Cost)
        double k @unit(s); // Ogni quanto genera i pacchetti (VA)
        double m @unit(s); //Tempo di monitoraggio (Cost)

        @display("bgb=461,189");
    gates:
        output out[nDL];


    submodules:
        packetGenerator: PacketGenerator {
            @display("p=412.055,145.715;i=device/pc4");
        }
        linkSelector: LinkSelector {
            @display("p=302.045,79.13;i=abstract/router");
        }
        dataLink[nDL]: DataLink {
            @display("i=block/mac;p=186.245,79.13");
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
