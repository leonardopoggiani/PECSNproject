//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

package aerocomsystemproject.simulations;

import aerocomsystemproject.Aircraft;
import aerocomsystemproject.ControlTower;
import aerocomsystemproject.DataLink;


network Aerocom
{
    parameters:
<<<<<<< HEAD
        int nDL = default(2);
        @display("bgb=296,147");
=======
    int nDL = default(2);
>>>>>>> 343b348c6a25c26a8b3c24d332d768cd8a585b92
    //@display("bgb=355,202");
    submodules:
        aircraft[nDL]: Aircraft {
            //@display("p=18200.729,11600.464");
<<<<<<< HEAD
            @display("p=210,88");
        }
        dataLink[nDL]: DataLink {
            //@display("p=18200.729,11600.464");
            @display("p=144,30");
        }
        controlTower: ControlTower {
            //@display("p=39,73");
            @display("p=35,69");
        }


    connections:
        for i=0..nDL-1, for j=0..nDL-1 {
            aircraft[i].out[j] --> dataLink[j].in[i];
=======
        }
        dataLink[nDL]: DataLink {
            //@display("p=18200.729,11600.464");
        }
        controlTower: ControlTower {
            //@display("p=39,73");
        }
        
        
    connections:
        for i=0..nDL-1, for j=0..nDL-1 {
            aircraft[i].out[j]--> dataLink[j].in[i];
>>>>>>> 343b348c6a25c26a8b3c24d332d768cd8a585b92
        }
        for i=0..nDL-1 {
            dataLink[i].out --> controlTower.in[i];
        }
<<<<<<< HEAD

=======
        
>>>>>>> 343b348c6a25c26a8b3c24d332d768cd8a585b92
}
