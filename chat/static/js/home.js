function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const openBtn = document.getElementById("openBtn");
  const isOpen = sidebar.style.width === "250px";

  sidebar.style.width = isOpen ? "0px" : "250px";
  openBtn.style.display = isOpen ? "block" : "none";

}