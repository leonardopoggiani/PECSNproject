package aerocomsystemproject.simulations;

import aerocomsystemproject.Aircraft;
import aerocomsystemproject.ControlTower;
import aerocomsystemproject.DataLink;
import aerocomsystemproject.LinkSelector;
import aerocomsystemproject.PacketGenerator;


network Aerocom
{
    parameters:
        int nA = default(1);
        int nDL = default(2);

    submodules:
        aircraft[nA]: Aircraft {
        }
        controlTower: ControlTower {
            @display("p=35,69");
        }


    connections:
        for i=0..nA-1, for j=0..nDL-1 {
            aircraft[i].out[j] --> controlTower.in[(i*nDL)+j];
        }
        
}
