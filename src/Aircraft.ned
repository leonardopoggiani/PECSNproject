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
<<<<<<< HEAD
            @display("p=64,160");
=======
            @display("p=201,172");
>>>>>>> 343b348c6a25c26a8b3c24d332d768cd8a585b92
        }
        linkSelector: LinkSelector {


            @display("p=57,37");
        }
<<<<<<< HEAD
    connections:
        for j=0..nDL-1 {
            linkSelector.out[j] --> out[j];
        }

        packetGenerator.out --> linkSelector.in;
=======
>>>>>>> 343b348c6a25c26a8b3c24d332d768cd8a585b92
}
