package aerocomsystemproject.simulations;

import aerocomsystemproject.Aircraft;
import aerocomsystemproject.ControlTower;


network Aerocom
{
    parameters:
        int nA = default(1);
        int nDL = default(2);

    submodules:
        aircraft[nA]: Aircraft {
            @display("p=111,94;b=9,9,rect;is=vs;i=icons/airplane-computer-icons-symbol-cargo-plane-shipping-transportation-wings-icon-e4f948bcb753cc1fc65d0f1025153943");
        }
        controlTower: ControlTower {
            @display("p=329,233;i=device/antennatower");
        }


    connections:
        for i=0..nA-1, for j=0..nDL-1 {
            aircraft[i].out[j] --> controlTower.in[(i*nDL)+j];
        }

}
