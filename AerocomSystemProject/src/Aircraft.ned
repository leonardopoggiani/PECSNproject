package aerocomsystemproject;

module Aircraft
{
    parameters:
        int nDL = default(2);//(Cost)
        double k @unit(s); // Ogni quanto genera i pacchetti (VA)
        double m @unit(s); //Tempo di monitoraggio (Cost)
        int size @unit(byte);	//Dimensione dei pacchetti (Cost)
        int serving; //Interfaccia sulla quale sta trasmettendo (1..nDL)

    gates:
        output out[nDL];


    submodules:
        packetGenerator: PacketGenerator {
            @display("p=201,172");
        }
        linkSelector: LinkSelector {


            @display("p=57,37");
        }
}
