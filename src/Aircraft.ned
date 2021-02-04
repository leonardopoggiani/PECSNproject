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
        	@display("p=282,62");
        }
        linkSelector: LinkSelector {
        }
        dataLink[nDL]: DataLink {
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
