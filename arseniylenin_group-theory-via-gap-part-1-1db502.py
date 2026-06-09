import time
t0 = time.time()


%%bash
sudo apt-get install -y m4
wget -nv https://github.com/gap-system/gap/releases/download/v4.12.2/gap-4.12.2.tar.gz
tar xzf gap-4.12.2.tar.gz
cd gap-4.12.2
./configure
make


print( time.time() - t0)


t0 = time.time()


%%bash
cd /kaggle/working/gap-4.12.2/pkg
../bin/BuildPackages.sh


print( time.time() - t0)


%%writefile gap_code.g
Display(1);
QuitGap();


%%time
!head "gap_code.g"



%%time
!./gap-4.12.2/bin/gap.sh -b gap_code.g





%%writefile gap_code1.g
pnck := Group((1,2), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
);
f := FreeGroup("1", "2");
hom := GroupHomomorphismByImages(f, pnck, GeneratorsOfGroup(f), GeneratorsOfGroup(pnck));
r := PermList([2, 1, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3] );
pre := PreImagesRepresentative(hom, r);
Display(r);
Display(pre);
Display(Length(pre));
QuitGap();


%%time
!./gap-4.12.2/bin/gap.sh -b gap_code1.g


'''%%writefile gap_code.g
pnck := Group((1,2), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26));
f := FreeGroup("1", "2");
hom := GroupHomomorphismByImages(f, pnck, GeneratorsOfGroup(f), GeneratorsOfGroup(pnck));
x := 0;
tmp_lst := [];

#for i in [1..1]do
    r := PseudoRandom(pnck);
    pre := PreImagesRepresentative(hom, r);
    Add(tmp_lst, Length(pre));  
    x := x + Length(pre);
#od;
Display(tmp_lst);

QuitGap();
%%time
!./gap-4.12.2/bin/gap.sh -b gap_code.g'''





#%%writefile gap_code.g
#pnck := Group((1,2), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15));
#f := FreeGroup("1", "2");
#hom := GroupHomomorphismByImages(f, pnck, GeneratorsOfGroup(f), GeneratorsOfGroup(pnck));
#r := Random(pnck);
#pre := PreImagesRepresentative(hom, r);
#Display(Length(pre));
#QuitGap();


#%%time
#!./gap-4.12.2/bin/gap.sh -b gap_code.g







