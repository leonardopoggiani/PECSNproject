package aerocomsystemproject.simulations;

import aerocomsystemproject.ControlTower;
import aerocomsystemproject.Aircraft;

network Aerocom
{
    parameters:
        int nA = default(1);
        int nDL = default(2);

    submodules:
        aircraft[nA]: Aircraft {
            @display("p=111,94;b=9,9,rect;is=n;");
        }
        controlTower: ControlTower {
            @display("p=329,233;");
        }


    connections:
        for i=0..nA-1, for j=0..nDL-1 {
            aircraft[i].out[j] --> controlTower.in[(i*nDL)+j];
        }

}
