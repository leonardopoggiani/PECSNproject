package aerocomsystemproject;

simple DataLink
{
     double t @unit(s); //Ogni quanto controlla la capacità dei data-link (VA)
     double k @unit(s);
     //int actualCapacity; //Capacità attuale del DL (VA)
     int dimPoolMax; //Const
     int dimPoolMin; //Const
     double X;
     int nDL;
     string setCapacityDistribution = default("exponential");
     
     
     gates:
         input in[nDL];
         output out;
        
}
