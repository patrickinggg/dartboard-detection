public class Coordinate {
    int x;
    int y;

    public Coordinate(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    @Override
    public String toString() {
        return "Coordinate{" +
                "x=" + x +
                ", y=" + y +
                '}';
    }

    @Override
    public boolean equals(Object obj){
        if(obj == null){
            return false;
        }else {
            if(this.getClass() == obj.getClass()){
                Coordinate cor = (Coordinate) obj;
                if (this.x == cor.x) {
                    if (this.y == cor.y){
                        return true;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }

            }else{
                return false;
            }
        }

    }
}
