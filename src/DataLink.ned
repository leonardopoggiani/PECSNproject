package aerocomsystemproject;


simple DataLink
{
     double t @unit(s); //Ogni quanto controlla la capacità dei data-link (VA)
     int actualCapacity; //Capacità attuale del DL (VA)
     int dimPoolMax; //Const
     int dimPoolMin; //Const
     double X;
     int nDL;
     
     gates:
         input in[nDL];
         output out;
        
}
