package aerocomsystemproject.simulations;

import aerocomsystemproject.Aircraft;
import aerocomsystemproject.ControlTower;
import aerocomsystemproject.DataLink;
import aerocomsystemproject.LinkSelector;
import aerocomsystemproject.PacketGenerator;


network Aerocom
{
    parameters:
        int nDL = default(2);
        // @display("bgb=296,147");	

    	//@display("bgb=355,202");
    submodules:
        aircraft[nDL]: Aircraft {
            //@display("p=18200.729,11600.464");
            // @display("p=210,88");
        }
        dataLink[nDL]: DataLink {
            //@display("p=18200.729,11600.464");
            // @display("p=144,30");
        }
        controlTower: ControlTower {
            //@display("p=39,73");
            @display("p=35,69");
        }


    connections:
        for i=0..nDL-1, for j=0..nDL-1 {
            aircraft[i].out[j] --> dataLink[j].in[i];
        }
        
        for i=0..nDL-1 {
            dataLink[i].out --> controlTower.in[i];
        }
        
}
