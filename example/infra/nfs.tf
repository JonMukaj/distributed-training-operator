resource "google_compute_disk" "nfs_disk" {
 name = "nfs-disk"
 size = 50
 zone = var.zone
 type = "pd-standard"
}